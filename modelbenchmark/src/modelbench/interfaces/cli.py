"""Command line interface.

Examples
--------
    modelbenchmark validate
    modelbenchmark hardware
    modelbenchmark list --what testsets
    modelbenchmark run --models minicpm5-1b --samples 5
    modelbenchmark compare output/RUN_A output/RUN_B
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from modelbench import __version__
from modelbench.adapters.config.settings import PROJECT_ROOT, ProjectConfig, load_config
from modelbench.adapters.omlx.backend import OMLXBackend
from modelbench.application.plan import plan_overview


def _print_plan(cfg: ProjectConfig) -> None:
    plan = plan_overview(cfg)
    print("mode: performance")
    print(f"models ({len(plan['models'])}): {plan['models']}")
    scenarios = [item["label"] for item in plan["scenarios"]]
    print(f"scenarios ({len(scenarios)}): {scenarios}")
    print(
        f"samples_per_dataset: {plan['samples_label']}   "
        f"max_tokens: {plan['max_tokens']}   repeats: {plan['repeats']}   "
        f"cache_mode: {plan['cache_mode']}   warmup: {plan['warmup_requests']}"
    )
    print(f"context filter: {plan['context_filter']} (1K: 0-1.5K; others: N +/- 0.5K)")
    print(
        "output layout: output/<batch_id>_performance_<model>/ "
        "(one bundle per model; batch metadata under output/batches/)"
    )
    for group in plan["groups"]:
        context = group.get("context_length")
        bucket = f"{context // 1024}k" if context else "native"
        print(
            f"  cell: {group['scenario']} | {group['dataset']} | {bucket} | "
            f"{group['n']} sample(s)"
        )
    if plan["cells_estimate"] is not None:
        print(f"estimated cells: ~{plan['cells_estimate']}")
    if plan["request_estimate"] is not None:
        print(f"estimated measured requests: ~{plan['request_estimate']}")
    elif plan["context_windows"]:
        print(
            "estimated measured requests: unknown until the legacy tokenizer filter "
            "is applied"
        )
    else:
        print("estimated measured requests: all available samples (known after dataset load)")
    if plan["max_requests"]:
        print(f"max_requests cap: {plan['max_requests']}")


def build_config(args: argparse.Namespace) -> ProjectConfig:
    overrides: Dict[str, Any] = {}
    run_overrides: Dict[str, Any] = {}
    perf_overrides: Dict[str, Any] = {}
    if getattr(args, "repeats", None) is not None:
        run_overrides["repeats"] = args.repeats
    if getattr(args, "cache_mode", None):
        run_overrides["cache_mode"] = args.cache_mode
    if getattr(args, "thinking_mode", None):
        run_overrides["thinking_mode"] = args.thinking_mode
    if getattr(args, "output_dir", None):
        run_overrides["output_dir"] = args.output_dir
    if getattr(args, "max_requests", None) is not None:
        run_overrides["max_requests"] = args.max_requests
    if getattr(args, "warmup", None) is not None:
        run_overrides["warmup_requests"] = args.warmup
    if getattr(args, "samples", None) is not None:
        perf_overrides["samples_per_dataset"] = args.samples
    if getattr(args, "max_tokens", None) is not None:
        perf_overrides["max_tokens"] = args.max_tokens
    if getattr(args, "context_windows", None) is not None:
        perf_overrides["context_windows"] = list(args.context_windows)
    if run_overrides:
        overrides["run"] = run_overrides
    if perf_overrides:
        overrides["perf"] = perf_overrides

    cfg = load_config(
        config_path=Path(args.config) if getattr(args, "config", None) else None,
        overrides=overrides,
    )

    # Optional named test set (dataset/test/<name>/testset.yaml).
    testset_name = getattr(args, "testset", None)
    if testset_name:
        from modelbench.adapters.config.testsets import load_testsets
        from modelbench.application.plan import apply_testset

        testsets = load_testsets(Path(cfg.datasets.testset_dir))
        if testset_name not in testsets:
            raise ValueError(
                f"unknown testset '{testset_name}'; available: {', '.join(sorted(testsets))}"
            )
        apply_testset(cfg, testsets[testset_name])

    if getattr(args, "models", None):
        cfg.selection.models = list(args.models)
    if getattr(args, "scenarios", None):
        cfg.selection.scenarios = list(args.scenarios)
    return cfg


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_run(args: argparse.Namespace) -> int:
    cfg = build_config(args)
    _print_plan(cfg)
    if args.dry_run:
        return 0

    from modelbench.application.batch import BatchRunner

    api_key = cfg.resolve_api_key()
    if not api_key:
        print(
            "ERROR: no oMLX API key. Set OMLX_API_KEY or configure server.api_key.",
            file=sys.stderr,
        )
        return 2
    client = OMLXBackend(
        base_url=cfg.server.base_url,
        api_key=api_key,
        admin_api_key=cfg.resolve_admin_api_key(),
        request_timeout_s=cfg.server.request_timeout_s,
        connect_timeout_s=cfg.server.connect_timeout_s,
        verify_ssl=cfg.server.verify_ssl,
        pid_file=cfg.server.pid_file,
    )
    try:
        probe = client.probe()
        if not probe.get("ok_health"):
            print(f"ERROR: oMLX not reachable: {probe.get('health_error')}", file=sys.stderr)
            return 3
        if not probe.get("ok_auth"):
            print(f"ERROR: API key rejected: {probe.get('models_error')}", file=sys.stderr)
            return 3
        # Preflight the selected models against the live catalog: a stale
        # models.yaml would otherwise 404 in the middle of a multi-model run.
        try:
            server_ids = {str(item.get("id")) for item in client.list_models()}
            missing = [
                model.id
                for model in cfg.selected_models()
                if model.id not in server_ids
            ]
        except Exception as exc:  # noqa: BLE001 - let the run itself report later
            print(f"WARNING: cannot verify models against /v1/models: {exc}", file=sys.stderr)
        else:
            if missing:
                print(
                    "ERROR: selected models are not on the oMLX server "
                    f"(sync models.yaml / re-download): {missing}",
                    file=sys.stderr,
                )
                return 3
        summary = BatchRunner(
            cfg,
            client,
            Path(cfg.run.output_dir),
            batch_id=args.run_id,
            overwrite=args.overwrite,
        ).run()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        client.close()


def _verify_dataset_manifest(raw_dir: Path) -> List[str]:
    """Verify every file recorded in a dataset's download manifest.

    Datasets without a manifest (e.g. longbench_v2, synthetic) are skipped;
    adapters still validate their own files when they load.
    """
    manifest_path = raw_dir / "download_manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{manifest_path}: unreadable manifest ({exc})"]
    files = manifest.get("files") or {}
    problems: List[str] = []
    for rel, info in files.items():
        path = raw_dir / rel
        if not path.is_file():
            problems.append(f"missing {rel}")
            continue
        expected_size = info.get("size")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            problems.append(
                f"size mismatch {rel}: {path.stat().st_size} != {expected_size}"
            )
            continue
        expected_sha = info.get("sha256")
        if expected_sha:
            digest = _sha256_file(path)
            if digest != expected_sha:
                problems.append(f"sha256 mismatch {rel}: {digest} != {expected_sha}")
    return problems


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cmd_validate(args: argparse.Namespace) -> int:
    failures: List[str] = []
    print(f"modelbenchmark v{__version__} — validation")
    print(f"project root: {PROJECT_ROOT}")

    # 1) imports
    modules = [
        "modelbench.domain.models",
        "modelbench.adapters.config.settings",
        "modelbench.adapters.omlx.client",
        "modelbench.adapters.omlx.cache",
        "modelbench.adapters.omlx.environment",
        "modelbench.adapters.host.sampler",
        "modelbench.adapters.host.hardware",
        "modelbench.adapters.datasets.registry",
        "modelbench.domain.evaluators.base",
        "modelbench.domain.scenarios.registry",
        "modelbench.domain.metrics",
        "modelbench.application.benchmark",
        "modelbench.adapters.storage.writer",
        "modelbench.adapters.storage.excel_report",
        "modelbench.interfaces.analysis.compare",
        "modelbench.interfaces.web.sample_worker",
    ]
    for module in modules:
        try:
            __import__(module)
            print(f"  [ok] import {module}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] import {module}: {exc}")
            failures.append(f"import {module}")

    # 2) config + performance plan
    try:
        cfg = build_config(args)
        print(
            f"  [ok] config: models={len(cfg.models)} scenarios={len(cfg.scenarios)} "
            f"samples_per_dataset={cfg.perf.samples_per_dataset or 'all'} "
            f"max_tokens={cfg.perf.max_tokens}"
        )
        # Enforce the default-config / MoE-only loading contract.
        for model in cfg.models:
            for key in model.load.settings:
                if key != "moe_expert_offload_enabled":
                    raise ValueError(f"model {model.id}: forbidden load setting {key}")
        print("  [ok] model loading: oMLX defaults (MoE offload only)")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] config: {exc}")
        return 1

    # 2b) testsets (root comes from datasets.testset_dir)
    try:
        from modelbench.adapters.config.testsets import load_testsets

        testsets = load_testsets(Path(cfg.datasets.testset_dir))
        print(f"  [ok] testsets: {', '.join(sorted(testsets))}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] testsets: {exc}")
        failures.append("testsets")

    # 3) datasets referenced by configured scenarios.
    #    Deduplicate by (dataset, dataset_params): the eight performance
    #    scenarios share `perf_jsonl` but select different scenario slices, so
    #    keying on the dataset name alone would validate only the first one.
    from modelbench.adapters.datasets.registry import get_adapter

    checked: Dict[str, int] = {}
    total_samples = 0
    for name, spec in cfg.scenarios.items():
        params = dict(spec.params.get("dataset_params") or {})
        key = spec.dataset + "|" + json.dumps(params, sort_keys=True)
        slice_name = params.get("scenario") or params.get("scenario_name")
        label = f"{spec.dataset}[{slice_name}]" if slice_name else spec.dataset
        if key in checked:
            continue
        try:
            adapter = get_adapter(
                spec.dataset,
                raw_dir=Path(cfg.datasets.raw_dir),
                processed_dir=Path(cfg.datasets.processed_dir),
                params=params,
                testset_dir=Path(cfg.datasets.testset_dir),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] dataset {label} ({name}): {exc}")
            failures.append(f"dataset {label}")
            continue
        if not args.no_checksum:
            raw_dir = Path(cfg.datasets.raw_dir) / spec.dataset
            problems = _verify_dataset_manifest(raw_dir)
            if problems:
                for problem in problems:
                    print(f"  [FAIL] dataset {label} checksum: {problem}")
                failures.append(f"dataset {label} checksum")
        try:
            samples = adapter.load()
            checked[key] = len(samples)
            total_samples += len(samples)
            print(f"  [ok] dataset {label}: {len(samples)} samples")
        except FileNotFoundError as exc:
            if args.allow_missing:
                print(f"  [warn] dataset {label} not downloaded: {exc}")
            else:
                print(f"  [FAIL] dataset {label} not downloaded: {exc}")
                failures.append(f"dataset {label} missing")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] dataset {label}: {type(exc).__name__}: {exc}")
            failures.append(f"dataset {label}")
    if checked:
        print(
            f"  [ok] total: {total_samples} samples across "
            f"{len(checked)} dataset selection(s)"
        )


    # 4) server (unless --offline)
    if not args.offline:
        api_key = cfg.resolve_api_key()
        if not api_key:
            print("  [warn] no oMLX API key configured; skipping server probe")
        else:
            client = OMLXBackend(
                base_url=cfg.server.base_url,
                api_key=api_key,
                admin_api_key=cfg.resolve_admin_api_key(),
                request_timeout_s=60,
            )
            try:
                probe = client.probe()
                ok = probe.get("ok_health") and probe.get("ok_auth") and probe.get("ok_tokenize")
                print(f"  [{'ok' if ok else 'FAIL'}] oMLX probe: {json.dumps(probe, ensure_ascii=False)[:600]}")
                if not ok:
                    failures.append("omlx probe")
                else:
                    # models.yaml must match the live server catalog; otherwise a
                    # run silently plans cells for models that will 404 later.
                    try:
                        server_ids = {
                            str(item.get("id"))
                            for item in client.list_models()
                            if item.get("id")
                        }
                    except Exception as exc:  # noqa: BLE001 - probe already passed
                        print(f"  [warn] cannot list /v1/models: {exc}")
                    else:
                        configured = {model.id for model in cfg.models if model.enabled}
                        missing = sorted(configured - server_ids)
                        extra = sorted(server_ids - configured)
                        if missing:
                            print(
                                "  [warn] configured models missing on server "
                                f"(selecting them will fail): {missing}"
                            )
                        if extra:
                            print(
                                "  [warn] server models not in models.yaml "
                                f"(cannot be selected until added): {extra}"
                            )
                        if not missing and not extra:
                            print("  [ok] models.yaml matches /v1/models")
            finally:
                client.close()

    if failures:
        print(f"\nvalidation FAILED ({len(failures)} issue(s)): {failures}")
        return 1
    print("\nvalidation passed")
    return 0


def cmd_hardware(args: argparse.Namespace) -> int:
    from modelbench.adapters.host.hardware import collect_hardware_profile
    from modelbench.adapters.omlx.environment import collect_omlx_environment

    cfg = build_config(args)
    api_key = cfg.resolve_api_key()
    client = None
    if api_key:
        client = OMLXBackend(base_url=cfg.server.base_url, api_key=api_key)
    try:
        profile = collect_hardware_profile(client)
        payload: Dict[str, Any] = {"hardware": profile}
        if client is not None:
            payload["omlx"] = collect_omlx_environment(
                client, [m.id for m in cfg.models if m.enabled]
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        if client:
            client.close()
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    what = args.what
    cfg = build_config(args)
    if what in ("scenarios", "all"):
        from modelbench.domain.scenarios.registry import available_scenarios

        print("scenarios:")
        for name in available_scenarios():
            spec = cfg.scenarios.get(name)
            dataset = spec.dataset if spec else "?"
            print(f"  {name:<22} dataset={dataset}")
    if what in ("datasets", "all"):
        from modelbench.adapters.datasets.registry import available_datasets

        print("datasets:")
        for name in available_datasets():
            path = Path(cfg.datasets.raw_dir) / name
            print(f"  {name:<14} raw_dir={path} exists={path.exists()}")
    if what in ("testsets", "all"):
        from modelbench.adapters.config.testsets import available_testsets

        print("testsets:")
        for name in available_testsets():
            print(f"  {name}")
    if what in ("models", "all"):
        print("models (configured):")
        for model in cfg.models:
            settings = dict(model.load.settings)
            print(
                f"  {model.id:<60} alias={model.alias or '-':<24} "
                f"enabled={model.enabled} load_settings={settings or '{}'}"
            )
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    script = PROJECT_ROOT / "tool" / "download" / "download.py"
    if not script.is_file():
        print(f"ERROR: downloader not found: {script}", file=sys.stderr)
        return 1
    datasets = list(getattr(args, "datasets", []) or [])
    if getattr(args, "dataset", None):
        datasets.append(args.dataset)
    if not datasets and not getattr(args, "list", False):
        datasets = ["all"]
    cmd = [sys.executable, str(script), *datasets]
    for flag in ("force", "no_process", "json", "list"):
        if getattr(args, flag, False):
            cmd.append("--" + flag.replace("_", "-"))
    if getattr(args, "params", None):
        cmd += ["--params", args.params]
    if getattr(args, "report", None):
        cmd += ["--report", args.report]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_analyze(args: argparse.Namespace) -> int:
    from modelbench.interfaces.analysis.cli import main as analysis_main

    argv = list(args.runs)
    if args.metrics:
        argv += ["--metrics", *args.metrics]
    if args.json:
        argv.append("--json")
    return analysis_main(argv)


def cmd_web(args: argparse.Namespace) -> int:
    from modelbench.interfaces.web.server import main as web_main

    argv = ["--host", args.host, "--port", str(args.port)]
    if getattr(args, "no_open", False):
        argv.append("--no-open")
    if getattr(args, "idle_timeout", None):
        argv += ["--idle-timeout", str(args.idle_timeout)]
    if getattr(args, "allow_external", False):
        argv.append("--allow-external")
    return web_main(argv)


def cmd_index(args: argparse.Namespace) -> int:
    from modelbench.adapters.storage.catalog import rebuild_index

    cfg = build_config(args)
    root = Path(cfg.run.output_dir)
    index = rebuild_index(root)
    print(
        f"index rebuilt: {root / 'index.json'} "
        f"({index['n_runs']} runs, {index.get('n_batches', 0)} batches)"
    )
    return 0


def cmd_export_analysis(args: argparse.Namespace) -> int:
    from modelbench.adapters.storage.excel_report import generate_all, generate_for_batch

    cfg = build_config(args)
    root = Path(cfg.run.output_dir)
    try:
        if args.all:
            results = generate_all(root)
        else:
            if not args.batch_ids:
                print("ERROR: provide batch id(s) or --all", file=sys.stderr)
                return 2
            results = [generate_for_batch(root, batch_id) for batch_id in args.batch_ids]
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"generated": results}, ensure_ascii=False, indent=2))
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    from modelbench.adapters.storage.catalog import ensure_index

    cfg = build_config(args)
    root = Path(cfg.run.output_dir)
    index = ensure_index(root, force=args.rebuild)
    runs = index.get("runs") or []
    if args.json:
        print(json.dumps(index, ensure_ascii=False, indent=2))
        return 0
    if not runs:
        print(f"no runs under {root}")
        return 0
    by_id = {entry.get("run_id"): entry for entry in runs}
    batches = index.get("batches") or []
    print(
        "batches are grouped by one test action; each model has its own bundle "
        "(time + method + model)."
    )
    for batch in batches:
        print(
            f"\n[{batch['batch_id']}] method={batch.get('method')} "
            f"models={batch.get('n_models')} samples={batch.get('n_samples')} "
            f"duration={batch.get('duration_s')}s created={batch.get('created_at')}"
        )
        for model_entry in batch.get("models") or []:
            entry = by_id.get(model_entry.get("run_id")) or {}
            skipped = f" skipped={len(entry.get('skipped') or [])}" if entry.get("skipped") else ""
            print(
                f"  - {entry.get('model') or model_entry.get('model') or '-':<30} "
                f"run_id={entry.get('run_id')} "
                f"samples={entry.get('n_samples') or '-'} "
                f"duration={entry.get('duration_s') or '-'}s "
                f"size={entry.get('bytes_total_human') or '-'}{skipped}"
            )
    grouped_ids = {
        model_entry.get("run_id")
        for batch in batches
        for model_entry in batch.get("models") or []
    }
    legacy = [entry for entry in runs if entry.get("run_id") not in grouped_ids]
    if legacy:
        print("\nlegacy / ungrouped runs:")
        for entry in legacy:
            print(
                f"  - {entry.get('run_id'):<40} sample(s)={entry.get('n_samples') or '-'} "
                f"duration={entry.get('duration_s') or '-'}s "
                f"size={entry.get('bytes_total_human') or '-'}"
            )
    print(
        f"\nindex: {root / 'index.json'} "
        f"({len(runs)} runs, {len(batches)} batches)"
    )
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelbench",
        description="Local LLM performance benchmark harness for oMLX (TTFT/TG TPS/memory).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", help="path to default.yaml")
        p.add_argument(
            "--testset",
            help="test set under dataset/test/ (e.g. performance)",
        )

    p_run = sub.add_parser("run", help="execute a performance benchmark run")
    add_common(p_run)
    p_run.add_argument("--models", nargs="+", help="model ids/aliases (default: first enabled)")
    p_run.add_argument("--scenarios", nargs="+", help="only run these scenario names")
    p_run.add_argument(
        "--samples",
        type=int,
        help=(
            "samples per scenario (each scenario is one dataset slice; default: "
            "config perf.samples_per_dataset; empty = all). The legacy field name "
            "is samples_per_dataset but the value is applied per scenario."
        ),
    )
    p_run.add_argument("--max-tokens", type=int, help="generation length per sample")
    p_run.add_argument(
        "--context-windows",
        nargs="+",
        type=int,
        metavar="TOKENS",
        help=(
            "legacy/diagnostic sample filter by prompt tokens (union; 1K/1024 accepts "
            "0-1536 tokens, other N accepts N-512..N+512). The performance dataset "
            "should be selected by bucket instead (scenario/perf_jsonl params.buckets); "
            "omitted = all buckets, cells keep context_length"
        ),
    )
    p_run.add_argument(
        "--warmup",
        type=int,
        metavar="N",
        help=(
            "discarded per-cell preheat requests before measuring (default 0; "
            "the model-level warmup is run.warmup_per_model=1 and is separate)"
        ),
    )
    p_run.add_argument("--repeats", type=int, help="measured repeats per sample (default 1)")
    p_run.add_argument("--cache-mode", choices=["cold", "warm", "none"])
    p_run.add_argument(
        "--thinking-mode",
        choices=["auto", "disabled", "enabled"],
        help="model thinking behavior (default: config, disabled)",
    )
    p_run.add_argument("--output-dir", help="results directory")
    p_run.add_argument("--run-id", help="explicit run id (default: UTC timestamp)")
    p_run.add_argument("--max-requests", type=int, help="global cap on measured requests")
    p_run.add_argument("--overwrite", action="store_true", help="overwrite existing run dir")
    p_run.add_argument("--dry-run", action="store_true", help="print the plan only")
    p_run.set_defaults(func=cmd_run)

    p_val = sub.add_parser("validate", help="self-check imports, config, datasets and server")
    add_common(p_val)
    p_val.add_argument("--offline", action="store_true", help="skip the live oMLX probe")
    p_val.add_argument(
        "--allow-missing",
        action="store_true",
        help="treat a missing dataset as a warning instead of a failure",
    )
    p_val.add_argument(
        "--no-checksum",
        action="store_true",
        help="skip SHA-256 verification against dataset/raw/<dataset>/download_manifest.json",
    )
    p_val.set_defaults(func=cmd_validate)

    p_hw = sub.add_parser("hardware", help="print hardware + oMLX configuration snapshot")
    p_hw.add_argument("--config")
    p_hw.set_defaults(func=cmd_hardware)

    p_list = sub.add_parser("list", help="list scenarios/datasets/models/testsets")
    p_list.add_argument(
        "--what",
        choices=["scenarios", "datasets", "models", "testsets", "all"],
        default="all",
    )
    p_list.add_argument("--config")
    p_list.set_defaults(func=cmd_list)

    p_dl = sub.add_parser("download", help="download benchmark datasets")
    p_dl.add_argument("datasets", nargs="*", help="dataset names, or 'all'")
    p_dl.add_argument("--dataset", help="dataset name or 'all' (repeatable via positional form)")
    p_dl.add_argument("--params", help="JSON adapter params applied to requested datasets")
    p_dl.add_argument("--no-process", action="store_true", help="download only")
    p_dl.add_argument("--force", action="store_true")
    p_dl.add_argument("--json", action="store_true")
    p_dl.add_argument("--report", help="write JSON report to this path")
    p_dl.add_argument("--list", action="store_true", help="list registered datasets")
    p_dl.set_defaults(func=cmd_download)

    p_an = sub.add_parser("analyze", help="compare 2+ run summaries")
    p_an.add_argument("runs", nargs="+", help="run directories or summary.json paths")
    p_an.add_argument(
        "--metrics",
        nargs="+",
        help="metrics to compare (default: ttft_ms gen_tps tpot_ms e2e_ms peak_memory_gb input_context_tokens)",
    )
    p_an.add_argument("--json", action="store_true", help="emit summaries as JSON")
    p_an.set_defaults(func=cmd_analyze)

    p_export = sub.add_parser(
        "export-analysis",
        help="generate Excel result analysis (result_analysis/) for batch(es)",
    )
    p_export.add_argument("batch_ids", nargs="*", help="batch id(s)")
    p_export.add_argument("--all", action="store_true", help="regenerate for all batches")
    p_export.add_argument("--output-dir", help="results directory (default: config run.output_dir)")
    p_export.add_argument("--config")
    p_export.set_defaults(func=cmd_export_analysis)

    p_runs = sub.add_parser("runs", help="list run bundles from output/index.json")
    p_runs.add_argument("--config")
    p_runs.add_argument("--output-dir", help="results directory (default: config run.output_dir)")
    p_runs.add_argument("--rebuild", action="store_true", help="rebuild the catalog first")
    p_runs.add_argument("--json", action="store_true", help="emit the catalog as JSON")
    p_runs.set_defaults(func=cmd_runs)

    p_index = sub.add_parser("index", help="rebuild the output/index.json run catalog")
    p_index.add_argument("--config")
    p_index.add_argument("--output-dir", help="results directory (default: config run.output_dir)")
    p_index.set_defaults(func=cmd_index)

    p_web = sub.add_parser("web", help="launch the local web console (opens the browser)")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8765)
    p_web.add_argument(
        "--no-open",
        action="store_true",
        help="do not open the console in the default browser automatically",
    )
    p_web.add_argument(
        "--idle-timeout",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="auto-stop after N seconds without web requests when no job is running (0 = off)",
    )
    p_web.add_argument(
        "--allow-external",
        action="store_true",
        help="allow binding to a non-loopback host (console has no authentication)",
    )
    p_web.set_defaults(func=cmd_web)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

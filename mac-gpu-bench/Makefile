BIN := output/build/macgpu
SRC := $(wildcard src/*.swift)
REPORT := output/reports/REPORT.md

all: $(BIN)

$(BIN): $(SRC)
	@mkdir -p output/build
	@swiftc -O -whole-module-optimization $(SRC) -o $(BIN).tmp && mv $(BIN).tmp $(BIN)
	@echo "built $(BIN)"

# 唯一入口：全套测试（= ./macgpu）
run: $(BIN)
	@./macgpu

# 用最近一次结果重新生成报告
report:
	@python3 tools/report.py -o $(REPORT)

selfcheck: $(BIN)
	@./macgpu selfcheck

# 文档与实现一致性（口径表、JSON 字段、目录约定）
docs:
	@python3 tools/docs_contract.py

test:
	@bash tests/smoke.sh

# 只清编译产物；结果与报告保留
clean:
	@rm -rf output/build
	@echo "cleaned output/build"

# 连结果和报告一起清
distclean: clean
	@rm -rf output/results output/reports
	@echo "cleaned output/"

.PHONY: all run report selfcheck docs test clean distclean

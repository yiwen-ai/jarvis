# options
ignore_output = &> /dev/null

.PHONY: run-dev test build docker

run-dev:
	@CONFIG_FILE_PATH=./config.toml cargo run

test:
	@cargo test -- --nocapture --include-ignored

lint:
	@cargo clippy --all-targets --all-features

fix:
	@cargo clippy --fix --bin "jarvis" --tests

docker:
	@docker build -t yiwen-ai/jarvis:latest .

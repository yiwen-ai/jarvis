# options
ignore_output = &> /dev/null

.PHONY: run-dev test build docker

run-dev:
	@CONFIG_FILE_PATH=./config.toml cargo run

test:
	@cargo test -- --nocapture

clippy:
	@cargo clippy --all-targets --all-features -- -D warnings

build:
	@cargo build --target x86_64-unknown-linux-gnu --release
	@cargo build --target aarch64-unknown-linux-gnu --release

docker:
	@docker buildx build --platform linux/amd64,linux/arm64 -t yiwen-ai/jarvis:latest .

FROM rust:1.85-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock clippy.toml /app/
COPY src /app/src
RUN cargo build --release --locked --bins

FROM debian:bookworm-slim

ENV LODE_DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/lode /usr/local/bin/lode
COPY --from=builder /app/target/release/loded /usr/local/bin/loded

EXPOSE 7979
CMD ["loded", "--host", "0.0.0.0", "--port", "7979"]

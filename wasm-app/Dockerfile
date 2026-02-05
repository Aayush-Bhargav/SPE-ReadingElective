# # STAGE 1: BUILDER
# FROM rust:1-slim-bookworm AS builder

# WORKDIR /usr/src/app

# # Install build tools
# RUN apt-get update && apt-get install -y \
#     pkg-config \
#     libssl-dev \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Add Target
# RUN rustup target add wasm32-wasip1

# # Install Spin CLI
# RUN curl -fsSL https://developer.fermyon.com/downloads/install.sh | bash -s -- -v v2.7.0 \
#     && mv spin /usr/local/bin/

# # Prepare Project
# COPY wasm-app/Cargo.toml .
# COPY wasm-app/spin.toml .
# RUN mkdir src && echo "fn main() {}" > src/lib.rs

# # Build Deps
# RUN spin build || true

# # Real Build
# COPY wasm-app/src ./src
# RUN touch src/lib.rs
# RUN spin build

# # STAGE 2: RUNNER
# FROM debian:bookworm-slim

# WORKDIR /app

# # Runtime deps
# RUN apt-get update && apt-get install -y \
#     curl \
#     ca-certificates \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# # Install Spin
# RUN curl -fsSL https://developer.fermyon.com/downloads/install.sh | bash -s -- -v v2.7.0 \
#     && mv spin /usr/local/bin/

# # Copy Artifacts
# COPY --from=builder /usr/src/app/target/wasm32-wasip1/release/wasm_app.wasm /app/wasm_app.wasm
# COPY models/mobilenetv2.onnx /app/mobilenetv2.onnx

# # [THE FIX] Copy spin.toml and PATCH it to point to the local file
# COPY wasm-app/spin.toml /app/spin.toml
# RUN sed -i 's|target/wasm32-wasip1/release/||g' /app/spin.toml

# # Run
# CMD ["spin", "up", "--listen", "0.0.0.0:8082"]

# ==========================================
# STAGE 1: BUILDER (Heavy)
# ==========================================
FROM rust:1-slim-bookworm AS builder

WORKDIR /usr/src/app

# Install tools needed for building
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Add WASM Target
RUN rustup target add wasm32-wasip1

# Install Spin CLI (v2.7.0)
RUN curl -fsSL https://developer.fermyon.com/downloads/install.sh | bash -s -- -v v2.7.0 \
    && mv spin /usr/local/bin/

# Prepare Project
COPY wasm-app/Cargo.toml .
COPY wasm-app/spin.toml .
RUN mkdir src && echo "fn main() {}" > src/lib.rs

# Cache Dependencies
RUN spin build || true

# Real Build
COPY wasm-app/src ./src
RUN touch src/lib.rs
RUN spin build

# ==========================================
# STAGE 2: RUNNER (Ultra-Light)
# ==========================================
FROM debian:bookworm-slim

WORKDIR /app

# 1. Runtime Dependencies
# We strictly only need SSL certificates for HTTPS support.
# NO git, NO curl, NO compilers.
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. [THE TRICK] Copy Spin Binary from Builder
# We don't run the installer script here, so we don't need git.
COPY --from=builder /usr/local/bin/spin /usr/local/bin/spin

# 3. Copy Artifacts
COPY --from=builder /usr/src/app/target/wasm32-wasip1/release/wasm_app.wasm /app/wasm_app.wasm
COPY models/mobilenetv2.onnx /app/mobilenetv2.onnx

# 4. Config
COPY wasm-app/spin.toml /app/spin.toml
RUN sed -i 's|target/wasm32-wasip1/release/||g' /app/spin.toml

# 5. Run
CMD ["spin", "up", "--listen", "0.0.0.0:8082"]
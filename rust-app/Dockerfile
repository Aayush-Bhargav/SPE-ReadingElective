# ===================================================
# STAGE 1: THE BUILDER (Ubuntu 24.04)
# We use Ubuntu because it has the newest GLIBC (2.39)
# required by the latest AI libraries.
# ===================================================
FROM ubuntu:24.04 as builder

WORKDIR /usr/src/app

# 1. Install System Tools (Curl, C++ Compiler, SSL)
# DEBIAN_FRONTEND=noninteractive prevents timezone popups
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Rust Manually (Since we aren't using the rust image)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# 3. Cache Dependencies (The "Dummy Build" Trick)
COPY rust-app/Cargo.toml .
COPY rust-app/Cargo.lock .
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release

# 4. Build the Real App
COPY rust-app/src ./src
# Touch main.rs to force a re-compile
RUN touch src/main.rs
RUN cargo build --release

# ===================================================
# STAGE 2: THE RUNNER (Ubuntu 24.04)
# We must use the SAME OS version for the runner
# so the system libraries match perfectly.
# ===================================================
# STAGE 2: THE RUNNER
FROM ubuntu:24.04

WORKDIR /app

# 1. Install Runtime Libraries (Changes Rarely)
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy the Model (Changes NEVER)
# We move this UP so it stays cached even if code changes.
COPY models/mobilenetv2.onnx /models/mobilenetv2.onnx

# 3. Copy the Binary (Changes OFTEN)
COPY --from=builder /usr/src/app/target/release/rust-app /app/rust-app

# 4. Run it
ENV LD_LIBRARY_PATH="/app:/usr/lib"
EXPOSE 8000
CMD ["./rust-app"]
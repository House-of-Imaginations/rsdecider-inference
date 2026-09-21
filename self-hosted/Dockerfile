# trixie: ort's prebuilt ONNX Runtime links against glibc >= 2.38 and GCC 13+ libstdc++
FROM rust:1-trixie AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock rustfmt.toml openapi.yaml ./
COPY src src
RUN cargo build --release && mkdir /out && cp target/release/rsdecider /out/ \
 && (cp target/release/*.so* /out/ 2>/dev/null || true)

FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/ /app/
ENV LD_LIBRARY_PATH=/app
WORKDIR /app
EXPOSE 3000 9000
ENTRYPOINT ["/app/rsdecider"]
CMD ["serve", "--config", "/etc/rsdecider/rsdecider.toml"]

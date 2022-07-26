use hello_wgpu::run;

fn main() {
    // Blocks main thread until future is ready (run loop exits).
    pollster::block_on(run());
}

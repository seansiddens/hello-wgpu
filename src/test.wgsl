struct VertexOutput {
    [[builtin(position)]] member: vec4<f32>;
};

var<private> texcoords: vec2<f32>;
var<private> gl_Position: vec4<f32>;
var<private> gl_VertexIndex: u32;

fn main_1() {
    var vertices: array<vec2<f32>,3u> = array<vec2<f32>,3u>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));

    let _e24 = gl_VertexIndex;
    let _e26 = vertices[_e24];
    gl_Position = vec4<f32>(_e26.x, _e26.y, f32(0), f32(1));
    let _e35 = gl_Position;
    texcoords = ((0.5 * _e35.xy) + vec2<f32>(0.5));
    return;
}

[[stage(vertex)]]
fn main([[builtin(vertex_index)]] param: u32) -> VertexOutput {
    gl_VertexIndex = param;
    main_1();
    let _e5 = gl_Position;
    return VertexOutput(_e5);
}

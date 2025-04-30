#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

struct PlanetData {
    vec3 position;
    vec3 velocity;
    vec3 color;
    float diameter;
    float mass;
};

layout(binding = 0) readonly buffer PlanetBuffer {
    PlanetData planets[];
};

layout(binding = 1) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    float deltaTime;
} ubo;

layout(location = 0) out vec3 fragColor;

void main() {
    vec3 pos = planets[gl_InstanceIndex].position;
    float scale = planets[gl_InstanceIndex].diameter;
    mat4 model = mat4(
        scale, 0.0, 0.0, 0.0,
        0.0, scale, 0.0, 0.0,
        0.0, 0.0, scale, 0.0,
        pos.x, pos.y, pos.z, 1.0
    );

    gl_Position = ubo.proj * ubo.view * model * vec4(inPosition, 1.0);
    fragColor = planets[gl_InstanceIndex].color;
}

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

const float G = 0.001;
const float C = 15.0;

void main() {
    vec3 pos = inPosition;
    float w = 0.0;
    float h = 60.0;

    for (uint i = 0; i < planets.length(); i++) {
        vec3 dir = planets[i].position - pos;
        float dist = length(dir);
        float rs = 2.0 * G * planets[i].mass / (C * C);
        if (dist < rs) dist = rs;
        if (dist > 1000) dist = rs;
        w += 2.0 * sqrt(rs * (dist - rs));
    }

    pos.z = w - h; 

    gl_Position = ubo.proj * ubo.view * vec4(pos, 1.0);

    fragColor = inColor;
}
import sys
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# GLSL Ray Tracing Shader Source
vertex_shader = """
#version 330 core
layout(location = 0) in vec3 position;
void main() {
    gl_Position = vec4(position, 1.0);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

uniform vec3 sphere_center;
uniform float sphere_radius;
uniform vec3 light_pos;
uniform vec3 sphere_color;
uniform vec3 bg_color;
uniform vec3 camera_pos;
uniform float camera_zoom;

// Ray-sphere intersection
bool intersectSphere(vec3 ro, vec3 rd, out float t) {
    vec3 oc = ro - sphere_center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - sphere_radius * sphere_radius;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    t = -b - h;
    if (t < 0.0) t = -b + h;
    return t > 0.0;
}

// Ray-pyramid intersection (regular pyramid, base at y=-0.7, apex at y=0.3, centered at (0.5, -0.2, 0.0))
bool intersectPyramid(vec3 ro, vec3 rd, out float t, out vec3 normal) {
    // Pyramid parameters (ขวา)
    vec3 apex = vec3(0.5, 0.1, 0.0);
    float base_y = -0.4;
    float base_size = 0.4;
    vec3 base_corners[4];
    base_corners[0] = vec3(0.3, base_y, -0.2);
    base_corners[1] = vec3(0.7, base_y, -0.2);
    base_corners[2] = vec3(0.7, base_y, 0.2);
    base_corners[3] = vec3(0.3, base_y, 0.2);
    float t_min = 1e9;
    bool hit = false;
    vec3 n_hit = vec3(0.0);
    // 4 side faces
    for (int i = 0; i < 4; ++i) {
        vec3 v0 = base_corners[i];
        vec3 v1 = base_corners[(i+1)%4];
        vec3 v2 = apex;
        vec3 e1 = v1 - v0;
        vec3 e2 = v2 - v0;
        vec3 pvec = cross(rd, e2);
        float det = dot(e1, pvec);
        if (abs(det) < 1e-6) continue;
        float invDet = 1.0 / det;
        vec3 tvec = ro - v0;
        float u = dot(tvec, pvec) * invDet;
        if (u < 0.0 || u > 1.0) continue;
        vec3 qvec = cross(tvec, e1);
        float v = dot(rd, qvec) * invDet;
        if (v < 0.0 || u + v > 1.0) continue;
        float t_face = dot(e2, qvec) * invDet;
        if (t_face > 0.001 && t_face < t_min) {
            t_min = t_face;
            n_hit = normalize(cross(e2, e1));
            hit = true;
        }
    }
    // Base face
    vec3 n_base = vec3(0,1,0);
    float denom = dot(rd, n_base);
    if (abs(denom) > 1e-6) {
        float t_base = (base_y - ro.y) / rd.y;
        if (t_base > 0.001 && t_base < t_min) {
            vec3 p = ro + rd * t_base;
            if (p.x > 0.3 && p.x < 0.7 && p.z > -0.2 && p.z < 0.2) {
                t_min = t_base;
                n_hit = n_base;
                hit = true;
            }
        }
    }
    t = t_min;
    normal = n_hit;
    return hit;
}

// Ray-pyramid intersection (พีระมิด 3 ขนาดเล็ก ติดพื้น ซ้ายสุดและข้างหน้า)
bool intersectPyramid3(vec3 ro, vec3 rd, out float t, out vec3 normal) {
    // Pyramid parameters (ขนาดเล็ก, ติดพื้น, ซ้ายสุด, ข้างหน้า)
    vec3 apex = vec3(-0.7, -0.4 + 0.22, 0.6); // apex ซ้ายกว่าเดิม, ข้างหน้า, ติดพื้น (base_y+0.22)
    float base_y = -0.4;
    float base_size = 0.18;
    vec3 base_corners[4];
    base_corners[0] = vec3(-0.79, base_y, 0.51); // x-0.79, z+0.51
    base_corners[1] = vec3(-0.61, base_y, 0.51); // x-0.61, z+0.51
    base_corners[2] = vec3(-0.61, base_y, 0.69); // x-0.61, z+0.69
    base_corners[3] = vec3(-0.79, base_y, 0.69); // x-0.79, z+0.69
    float t_min = 1e9;
    bool hit = false;
    vec3 n_hit = vec3(0.0);
    // 4 side faces
    for (int i = 0; i < 4; ++i) {
        vec3 v0 = base_corners[i];
        vec3 v1 = base_corners[(i+1)%4];
        vec3 v2 = apex;
        vec3 e1 = v1 - v0;
        vec3 e2 = v2 - v0;
        vec3 pvec = cross(rd, e2);
        float det = dot(e1, pvec);
        if (abs(det) < 1e-6) continue;
        float invDet = 1.0 / det;
        vec3 tvec = ro - v0;
        float u = dot(tvec, pvec) * invDet;
        if (u < 0.0 || u > 1.0) continue;
        vec3 qvec = cross(tvec, e1);
        float v = dot(rd, qvec) * invDet;
        if (v < 0.0 || u + v > 1.0) continue;
        float t_face = dot(e2, qvec) * invDet;
        if (t_face > 0.001 && t_face < t_min) {
            t_min = t_face;
            n_hit = normalize(cross(e2, e1));
            hit = true;
        }
    }
    // Base face
    vec3 n_base = vec3(0,1,0);
    float denom = dot(rd, n_base);
    if (abs(denom) > 1e-6) {
        float t_base = (base_y - ro.y) / rd.y;
        if (t_base > 0.001 && t_base < t_min) {
            vec3 p = ro + rd * t_base;
            if (p.x > -0.79 && p.x < -0.61 && p.z > 0.51 && p.z < 0.69) {
                t_min = t_base;
                n_hit = n_base;
                hit = true;
            }
        }
    }
    t = t_min;
    normal = n_hit;
    return hit;
}

// Ray-box intersection (axis-aligned box)
bool intersectBox(vec3 ro, vec3 rd, vec3 box_min, vec3 box_max, out float t, out vec3 normal) {
    vec3 inv_rd = 1.0 / rd;
    vec3 t0s = (box_min - ro) * inv_rd;
    vec3 t1s = (box_max - ro) * inv_rd;
    vec3 tsmaller = min(t0s, t1s);
    vec3 tbigger = max(t0s, t1s);
    float tmin = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
    float tmax = min(min(tbigger.x, tbigger.y), tbigger.z);
    if (tmax < 0.0 || tmin > tmax) return false;
    t = (tmin > 0.0) ? tmin : tmax;
    vec3 hit = ro + rd * t;
    float eps = 1e-4;
    if (abs(hit.x - box_min.x) < eps) normal = vec3(-1,0,0);
    else if (abs(hit.x - box_max.x) < eps) normal = vec3(1,0,0);
    else if (abs(hit.y - box_min.y) < eps) normal = vec3(0,-1,0);
    else if (abs(hit.y - box_max.y) < eps) normal = vec3(0,1,0);
    else if (abs(hit.z - box_min.z) < eps) normal = vec3(0,0,-1);
    else normal = vec3(0,0,1);
    return true;
}

// Ray-cylinder intersection (infinite vertical cylinder, capped)
bool intersectCylinder(vec3 ro, vec3 rd, vec3 center, float radius, float y_min, float y_max, out float t, out vec3 normal) {
    // Ignore y for side surface
    vec2 oc = ro.xz - center.xz;
    vec2 dir = rd.xz;
    float a = dot(dir, dir);
    float b = 2.0 * dot(oc, dir);
    float c = dot(oc, oc) - radius * radius;
    float h = b * b - 4.0 * a * c;
    t = 1e9;
    normal = vec3(0.0);
    bool hit = false;
    if (h >= 0.0) {
        h = sqrt(h);
        float t0 = (-b - h) / (2.0 * a);
        float t1 = (-b + h) / (2.0 * a);
        for (int i = 0; i < 2; ++i) {
            float t_side = (i == 0) ? t0 : t1;
            if (t_side > 0.001) {
                float y = ro.y + rd.y * t_side;
                if (y > y_min && y < y_max && t_side < t) {
                    t = t_side;
                    vec3 hit_pos = ro + rd * t;
                    normal = normalize(vec3(hit_pos.x - center.x, 0.0, hit_pos.z - center.z));
                    hit = true;
                }
            }
        }
    }
    // Cap intersections
    for (int cap = 0; cap < 2; ++cap) {
        float y_cap = (cap == 0) ? y_min : y_max;
        if (abs(rd.y) > 1e-6) {
            float t_cap = (y_cap - ro.y) / rd.y;
            if (t_cap > 0.001 && t_cap < t) {
                vec3 p = ro + rd * t_cap;
                if (length(p.xz - center.xz) < radius) {
                    t = t_cap;
                    normal = vec3(0, cap == 0 ? -1 : 1, 0);
                    hit = true;
                }
            }
        }
    }
    return hit;
}

// Simple reflection
vec3 reflectRay(vec3 I, vec3 N) {
    return I - 2.0 * dot(N, I) * N;
}

void main() {
    // Camera setup
    vec3 ro = camera_pos;
    vec2 uv = (gl_FragCoord.xy / vec2(800, 600)) * 2.0 - 1.0;
    uv.x *= 800.0/600.0;
    uv *= camera_zoom;
    vec3 rd = normalize(vec3(uv, -1.5)); // Ray direction

    // กล่อง: ตำแหน่งกลาง (0.0, -0.7, 1.0), ขนาดเล็ก (y -0.7 ถึง -0.54)
    vec3 box_min = vec3(-0.08, -0.7, 0.90);
    vec3 box_max = vec3(0.08, -0.54, 1.10);

    float t_sphere, t_pyr, t_box, t_cyl;
    vec3 n_pyr, n_box, n_cyl;
    bool hit_sphere = intersectSphere(ro, rd, t_sphere);
    bool hit_pyr = intersectPyramid(ro, rd, t_pyr, n_pyr);
    bool hit_box = intersectBox(ro, rd, box_min, box_max, t_box, n_box);
    // ทรงกระบอก: ติดพื้น, ขวาของพีระมิด (center = (1.0, -0.7, 0.0)), r = 0.18, h = 0.64
    vec3 cyl_center = vec3(1.0, -0.7, 0.0);
    float cyl_radius = 0.18;
    float cyl_ymin = -0.7;
    float cyl_ymax = -0.06;
    bool hit_cyl = intersectCylinder(ro, rd, cyl_center, cyl_radius, cyl_ymin, cyl_ymax, t_cyl, n_cyl);
    bool use_sphere = hit_sphere && (!hit_pyr || t_sphere < t_pyr) && (!hit_box || t_sphere < t_box) && (!hit_cyl || t_sphere < t_cyl);
    bool use_pyr = hit_pyr && (!hit_sphere || t_pyr < t_sphere) && (!hit_box || t_pyr < t_box) && (!hit_cyl || t_pyr < t_cyl);
    bool use_box = hit_box && (!hit_sphere || t_box < t_sphere) && (!hit_pyr || t_box < t_pyr) && (!hit_cyl || t_box < t_cyl);
    bool use_cyl = hit_cyl && (!hit_sphere || t_cyl < t_sphere) && (!hit_pyr || t_cyl < t_pyr) && (!hit_box || t_cyl < t_box);
    float t_pyr3;
    vec3 n_pyr3;
    bool hit_pyr3 = intersectPyramid3(ro, rd, t_pyr3, n_pyr3);
    bool use_pyr3 = hit_pyr3 && (!use_sphere || t_pyr3 < t_sphere) && (!use_pyr || t_pyr3 < t_pyr) && (!use_box || t_pyr3 < t_box) && (!use_cyl || t_pyr3 < t_cyl);
    if (use_sphere) {
        float t = t_sphere;
        vec3 hit = ro + rd * t;
        vec3 normal = normalize(hit - sphere_center);
        vec3 light_dir = normalize(light_pos - hit);
        // Phong shading
        float diff = max(dot(normal, light_dir), 0.0);
        float ambient = 0.18;
        // Specular (ลด exponent และน้ำหนัก)
        vec3 view_dir = normalize(ro - hit);
        vec3 reflect_dir = reflect(-light_dir, normal);
        float spec = 0.3 * pow(max(dot(view_dir, reflect_dir), 0.0), 24.0);
        // Soft shadow (sample 4 directions)
        float shadow = 0.0;
        for (int i = 0; i < 4; ++i) {
            float angle = 3.14159 * 0.5 * float(i);
            vec3 offset = 0.02 * (cos(angle) * normal + sin(angle) * cross(normal, light_dir));
            float t_shadow;
            if (intersectSphere(hit + normal * 0.001 + offset, light_dir, t_shadow)) shadow += 0.25;
        }
        shadow = mix(1.0, 0.5, shadow); // soft shadow เนียนขึ้น
        // Reflection (simple, 1 bounce, ลดน้ำหนัก)
        vec3 refl_dir = reflectRay(rd, normal);
        float t_refl;
        vec3 refl_col = vec3(0.0);
        float t_plane = (-0.7 - hit.y) / refl_dir.y;
        bool hit_sphere = intersectSphere(hit + normal * 0.01, refl_dir, t_refl);
        bool hit_ground = (t_plane > 0.0);
        if (hit_sphere && (!hit_ground || t_refl < t_plane)) {
            vec3 hit2 = hit + refl_dir * t_refl;
            vec3 n2 = normalize(hit2 - sphere_center);
            refl_col = sphere_color * 0.7 + 0.3 * vec3(1.0);
        } else {
            refl_col = vec3(0.0); // ไม่สะท้อนพื้น
        }
        refl_col = mix(sphere_color, refl_col, 0.5); // blend reflection color for smoothness
        vec3 color = sphere_color * (ambient + diff * shadow) + spec * vec3(1.0) + 0.12 * refl_col;
        FragColor = vec4(color, 1.0);
    } else if (use_pyr) {
        vec3 hit = ro + rd * t_pyr;
        vec3 normal = n_pyr;
        vec3 light_dir = normalize(light_pos - hit);
        float diff = max(dot(normal, light_dir), 0.0);
        float ambient = 0.25; // เพิ่ม ambient ให้พีระมิดดูเหลืองขึ้น
        vec3 view_dir = normalize(ro - hit);
        vec3 reflect_dir = reflect(-light_dir, normal);
        float spec = 0.2 * pow(max(dot(view_dir, reflect_dir), 0.0), 16.0);
        // Soft shadow from sphere and pyramid (sample หลายทิศทาง)
        float shadow = 0.0;
        int shadow_samples = 8;
        for (int i = 0; i < shadow_samples; ++i) {
            float angle = 6.28318 * float(i) / float(shadow_samples);
            vec3 offset = 0.025 * (cos(angle) * normal + sin(angle) * cross(normal, light_dir));
            float t_shadow, t_shadow_pyr;
            vec3 n_dummy;
            bool shadow_sphere = intersectSphere(hit + normal * 0.001 + offset, light_dir, t_shadow) && t_shadow > 0.0;
            bool shadow_pyr = intersectPyramid(hit + normal * 0.001 + offset, light_dir, t_shadow_pyr, n_dummy) && t_shadow_pyr > 0.0 && t_shadow_pyr < length(light_pos - hit);
            if (shadow_sphere || shadow_pyr) shadow += 1.0;
        }
        shadow = 1.0 - shadow / float(shadow_samples); // soft shadow
        shadow = mix(1.0, 0.2, 1.0 - shadow); // blend เงาให้เข้มขึ้น
        vec3 pyr_color = vec3(1.0, 1.0, 0.2); // สีเหลือง
        vec3 color = pyr_color * (ambient + diff * shadow) + spec * vec3(1.0);
        FragColor = vec4(color, 1.0);
    } else if (use_pyr3) {
        vec3 hit = ro + rd * t_pyr3;
        vec3 normal = n_pyr3;
        vec3 light_dir = normalize(light_pos - hit);
        float diff = max(dot(normal, light_dir), 0.0);
        float ambient = 0.23;
        vec3 view_dir = normalize(ro - hit);
        vec3 reflect_dir = reflect(-light_dir, normal);
        float spec = 0.18 * pow(max(dot(view_dir, reflect_dir), 0.0), 14.0);
        // Soft shadow (sample หลายทิศทาง, ตรวจสอบทุกวัตถุ)
        float shadow = 0.0;
        int shadow_samples = 8;
        for (int i = 0; i < shadow_samples; ++i) {
            float angle = 6.28318 * float(i) / float(shadow_samples);
            vec3 offset = 0.025 * (cos(angle) * normal + sin(angle) * cross(normal, light_dir));
            float t_shadow, t_shadow_pyr, t_shadow_pyr3, t_shadow_box, t_shadow_cyl = 0.0;
            vec3 n_dummy;
            bool shadow_sphere = intersectSphere(hit + normal * 0.001 + offset, light_dir, t_shadow) && t_shadow > 0.0;
            bool shadow_pyr = intersectPyramid(hit + normal * 0.001 + offset, light_dir, t_shadow_pyr, n_dummy) && t_shadow_pyr > 0.0 && t_shadow_pyr < length(light_pos - hit);
            bool shadow_pyr3 = intersectPyramid3(hit + normal * 0.001 + offset, light_dir, t_shadow_pyr3, n_dummy) && t_shadow_pyr3 > 0.0 && t_shadow_pyr3 < length(light_pos - hit);
            bool shadow_box = intersectBox(hit + normal * 0.001 + offset, light_dir, box_min, box_max, t_shadow_box, n_dummy) && t_shadow_box > 0.0 && t_shadow_box < length(light_pos - hit);
            bool shadow_cyl = intersectCylinder(hit + normal * 0.001 + offset, light_dir, cyl_center, cyl_radius, cyl_ymin, cyl_ymax, t_shadow_cyl, n_dummy) && t_shadow_cyl > 0.0 && t_shadow_cyl < length(light_pos - hit);
            if (shadow_sphere || shadow_pyr || shadow_pyr3 || shadow_box || shadow_cyl) shadow += 1.0;
        }
        shadow = 1.0 - shadow / float(shadow_samples);
        shadow = mix(1.0, 0.2, 1.0 - shadow);
        vec3 pyr3_color = vec3(0.2, 1.0, 0.3); // สีเขียว
        vec3 color = pyr3_color * (ambient + diff * shadow) + spec * vec3(1.0);
        FragColor = vec4(color, 1.0);
    } else if (use_box) {
        vec3 hit = ro + rd * t_box;
        vec3 normal = n_box;
        vec3 light_dir = normalize(light_pos - hit);
        float diff = max(dot(normal, light_dir), 0.0);
        float ambient = 0.22;
        vec3 view_dir = normalize(ro - hit);
        vec3 reflect_dir = reflect(-light_dir, normal);
        float spec = 0.25 * pow(max(dot(view_dir, reflect_dir), 0.0), 20.0);
        // Soft shadow (sample หลายทิศทาง, ตรวจสอบ sphere/pyramid/box)
        float shadow = 0.0;
        int shadow_samples = 8;
        for (int i = 0; i < shadow_samples; ++i) {
            float angle = 6.28318 * float(i) / float(shadow_samples);
            vec3 offset = 0.025 * (cos(angle) * normal + sin(angle) * cross(normal, light_dir));
            float t_shadow, t_shadow_pyr, t_shadow_box;
            vec3 n_dummy;
            bool shadow_sphere = intersectSphere(hit + normal * 0.001 + offset, light_dir, t_shadow) && t_shadow > 0.0;
            bool shadow_pyr = intersectPyramid(hit + normal * 0.001 + offset, light_dir, t_shadow_pyr, n_dummy) && t_shadow_pyr > 0.0 && t_shadow_pyr < length(light_pos - hit);
            bool shadow_box = intersectBox(hit + normal * 0.001 + offset, light_dir, box_min, box_max, t_shadow_box, n_dummy) && t_shadow_box > 0.0 && t_shadow_box < length(light_pos - hit);
            if (shadow_sphere || shadow_pyr || shadow_box) shadow += 1.0;
        }
        shadow = 1.0 - shadow / float(shadow_samples);
        shadow = mix(1.0, 0.3, 1.0 - shadow);
        vec3 box_color = vec3(0.2, 0.6, 1.0); // สีฟ้า
        vec3 color = box_color * (ambient + diff * shadow) + spec * vec3(1.0);
        FragColor = vec4(color, 1.0);
    } else if (use_cyl) {
        vec3 hit = ro + rd * t_cyl;
        vec3 normal = n_cyl;
        vec3 light_dir = normalize(light_pos - hit);
        float diff = max(dot(normal, light_dir), 0.0);
        float ambient = 0.22;
        vec3 view_dir = normalize(ro - hit);
        vec3 reflect_dir = reflect(-light_dir, normal);
        float spec = 0.22 * pow(max(dot(view_dir, reflect_dir), 0.0), 18.0);
        // Soft shadow (sample หลายทิศทาง, ตรวจสอบทุกวัตถุ)
        float shadow = 0.0;
        int shadow_samples = 8;
        for (int i = 0; i < shadow_samples; ++i) {
            float angle = 6.28318 * float(i) / float(shadow_samples);
            vec3 offset = 0.025 * (cos(angle) * normal + sin(angle) * cross(normal, light_dir));
            float t_shadow, t_shadow_pyr, t_shadow_box, t_shadow_cyl = 0.0;
            vec3 n_dummy;
            bool shadow_sphere = intersectSphere(hit + normal * 0.001 + offset, light_dir, t_shadow) && t_shadow > 0.0;
            bool shadow_pyr = intersectPyramid(hit + normal * 0.001 + offset, light_dir, t_shadow_pyr, n_dummy) && t_shadow_pyr > 0.0 && t_shadow_pyr < length(light_pos - hit);
            bool shadow_box = intersectBox(hit + normal * 0.001 + offset, light_dir, box_min, box_max, t_shadow_box, n_dummy) && t_shadow_box > 0.0 && t_shadow_box < length(light_pos - hit);
            bool shadow_cyl = intersectCylinder(hit + normal * 0.001 + offset, light_dir, cyl_center, cyl_radius, cyl_ymin, cyl_ymax, t_shadow_cyl, n_dummy) && t_shadow_cyl > 0.0 && t_shadow_cyl < length(light_pos - hit);
            if (shadow_sphere || shadow_pyr || shadow_box || shadow_cyl) shadow += 1.0;
        }
        shadow = 1.0 - shadow / float(shadow_samples);
        shadow = mix(1.0, 0.25, 1.0 - shadow);
        vec3 cyl_color = vec3(1.0, 0.5, 0.1); // สีส้ม
        vec3 color = cyl_color * (ambient + diff * shadow) + spec * vec3(1.0);
        FragColor = vec4(color, 1.0);
    } else {
        // Ground plane at y = -0.7
        float t_plane = ( -0.7 - ro.y ) / rd.y;
        if (t_plane > 0.0) {
            vec3 hit = ro + rd * t_plane;
            vec3 light_dir = normalize(light_pos - hit);
            float diff = max(dot(vec3(0,1,0), light_dir), 0.0);
            float ambient = 0.25;
            // Soft shadow from sphere (เพิ่ม sample เป็น 12 ทิศทาง)
            float shadow = 0.0;
            int shadow_samples = 12;
            for (int i = 0; i < shadow_samples; ++i) {
                float angle = 6.28318 * float(i) / float(shadow_samples);
                vec3 offset = 0.02 * (cos(angle) * vec3(0,1,0) + sin(angle) * vec3(1,0,0));
                float t_shadow, t_shadow_pyr, t_shadow_pyr3, t_shadow_box, t_shadow_cyl = 0.0;
                vec3 n_dummy;
                bool shadow_sphere = intersectSphere(hit + vec3(0,1,0) * 0.001 + offset, light_dir, t_shadow) && t_shadow > 0.0;
                bool shadow_pyr = intersectPyramid(hit + vec3(0,1,0) * 0.001 + offset, light_dir, t_shadow_pyr, n_dummy) && t_shadow_pyr > 0.0 && t_shadow_pyr < length(light_pos - hit);
                bool shadow_pyr3 = intersectPyramid3(hit + vec3(0,1,0) * 0.001 + offset, light_dir, t_shadow_pyr3, n_dummy) && t_shadow_pyr3 > 0.0 && t_shadow_pyr3 < length(light_pos - hit);
                bool shadow_box = intersectBox(hit + vec3(0,1,0) * 0.001 + offset, light_dir, box_min, box_max, t_shadow_box, n_dummy) && t_shadow_box > 0.0 && t_shadow_box < length(light_pos - hit);
                bool shadow_cyl = intersectCylinder(hit + vec3(0,1,0) * 0.001 + offset, light_dir, cyl_center, cyl_radius, cyl_ymin, cyl_ymax, t_shadow_cyl, n_dummy) && t_shadow_cyl > 0.0 && t_shadow_cyl < length(light_pos - hit);
                if (shadow_sphere || shadow_pyr || shadow_pyr3 || shadow_box || shadow_cyl) shadow += 1.0;
            }
            shadow = 1.0 - shadow / float(shadow_samples); // เงาเนียนขึ้น
            shadow = mix(1.0, 0.8, 1.0 - shadow); // blend เงาให้เนียน
            // Checkerboard pattern กลับมา
            float checker = mod(floor(hit.x*3.0)+floor(hit.z*3.0),2.0);
            // ทำให้สี checkerboard อ่อนลง
            vec3 ground_col = mix(bg_color, bg_color * 0.9 + vec3(1.0,1.0,1.0)*0.1, checker);
            float shade = ambient + diff * shadow;
            shade = mix(shade, ambient, 0.18);
            vec3 color = ground_col * shade;
            FragColor = vec4(color, 1.0);
        } else {
            // Sky gradient
            float t = 0.5 * (rd.y + 1.0);
            vec3 sky = mix(vec3(0.7,0.8,1.0), vec3(1.0,1.0,1.0), t);
            FragColor = vec4(sky, 1.0);
        }
    }
}
"""
# ฟังก์ชันสำหรับคอมไพล์ shader GLSL (vertex หรือ fragment)
# ใช้เมื่อเราต้องการสร้าง shader object จาก source code GLSL
def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    glShaderSource(shader, source)
    glCompileShader(shader)
    if glGetShaderiv(shader, GL_COMPILE_STATUS) != GL_TRUE:
        raise RuntimeError(glGetShaderInfoLog(shader))
    return shader

# ฟังก์ชันสำหรับสร้าง shader program
# ใช้เมื่อเราต้องการรวม vertex shader และ fragment shader เข้าด้วยกัน
def create_shader_program():
    vs = compile_shader(vertex_shader, GL_VERTEX_SHADER)
    fs = compile_shader(fragment_shader, GL_FRAGMENT_SHADER)
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        raise RuntimeError(glGetProgramInfoLog(program))
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program

# ฟังก์ชัน callback สำหรับวาดภาพ (render scene)
# จะถูกเรียกโดยอัตโนมัติเมื่อหน้าต่างต้องการวาดใหม่
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader_program)
    # Set uniforms
    glUniform3f(glGetUniformLocation(shader_program, 'sphere_center'), -0.5, 0.0, 0.0)
    glUniform1f(glGetUniformLocation(shader_program, 'sphere_radius'), 0.3)
    glUniform3f(glGetUniformLocation(shader_program, 'light_pos'), 2.0, 2.0, 2.0)
    glUniform3f(glGetUniformLocation(shader_program, 'sphere_color'), 1.0, 0.1, 0.1)
    glUniform3f(glGetUniformLocation(shader_program, 'bg_color'), 0.8, 0.8, 0.8)
    # Pass camera position and zoom as uniforms
    glUniform3f(glGetUniformLocation(shader_program, 'camera_pos'), *camera_pos)
    glUniform1f(glGetUniformLocation(shader_program, 'camera_zoom'), camera_zoom)
    # Draw fullscreen quad
    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
    glutSwapBuffers()

# ฟังก์ชัน callback สำหรับเปลี่ยนขนาดหน้าต่าง
# ใช้ปรับ viewport ให้ตรงกับขนาดใหม่ของหน้าต่าง
def reshape(width, height):
    glViewport(0, 0, width, height)

# --- Camera control state ---
camera_pos = [0.0, -0.5, 2.0]  # [x, y, z]
camera_zoom = 1.0


# ฟังก์ชัน callback สำหรับรับ input จากแป้นพิมพ์
# ใช้ควบคุมตำแหน่งกล้อง (pan) และซูมเข้า/ออก
# --- Keyboard controls for pan/zoom ---
def keyboard(key, x, y):
    global camera_pos, camera_zoom
    step = 0.08
    zoom_step = 0.1
    if key == b'w':
        camera_pos[2] -= step * camera_zoom  # forward
    elif key == b's':
        camera_pos[2] += step * camera_zoom  # backward
    elif key == b'a':
        camera_pos[0] -= step * camera_zoom  # left
    elif key == b'd':
        camera_pos[0] += step * camera_zoom  # right
    elif key == b'q':
        camera_pos[1] += step * camera_zoom  # up
    elif key == b'e':
        camera_pos[1] -= step * camera_zoom  # down
    elif key == b'+':
        camera_zoom = max(0.2, camera_zoom - zoom_step)  # zoom in
    elif key == b'-':
        camera_zoom += zoom_step  # zoom out
    glutPostRedisplay()


# ฟังก์ชันหลักสำหรับเริ่มต้นโปรแกรมและสร้างหน้าต่าง OpenGL
# ใช้สำหรับเซ็ตอัพทุกอย่างก่อนเข้าสู่ loop หลักของ GLUT
def main():
    global shader_program, vao
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b'Ray Tracing OpenGL')
    shader_program = create_shader_program()
    # Fullscreen quad
    quad = np.array([
        -1, -1, 0,
         1, -1, 0,
        -1,  1, 0,
         1,  1, 0
    ], dtype=np.float32)
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    glutDisplayFunc(display)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMainLoop()

# จุดเริ่มต้นโปรแกรม
if __name__ == '__main__':
    main()
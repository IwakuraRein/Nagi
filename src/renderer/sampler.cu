#include "sampler.cuh"

namespace nagi {

__device__ __host__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = hash((1 << 31) | (depth << 22) | iter) ^ hash(index);
    return thrust::default_random_engine(h);
}

__device__ __host__ glm::vec3 getDifferentDir(const glm::vec3& dir) {
    // Find a direction that is not the dir based of whether or not the
    // dir's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 T;
    if (fabsf(dir.x) < INV_SQRT_THREE) {
        T = glm::vec3(1, 0, 0);
    }
    else if (fabsf(dir.y) < INV_SQRT_THREE) {
        T = glm::vec3(0, 1, 0);
    }
    else {
        T = glm::vec3(0, 0, 1);
    }
    return T;
}

// reference: https://agraphicsguy.wordpress.com/2015/11/01/sampling-microfacet-brdf/
__device__ __host__ glm::vec3 nagi::ggxImportanceSampler(
    float alpha, const glm::vec3& wi, const glm::vec3& normal, float* pdf, float rnd1, float rnd2) {

    float a2 = alpha * alpha;
    float phi = rnd1 * TWO_PI;
    float cosTheta = glm::clamp(sqrtf((1.f - rnd2) / (rnd2 * (a2 - 1.f) + 1)), 0.f, 1.f);
    float cosTheta2 = cosTheta * cosTheta;
    float sinTheta = sqrtf(1.f - cosTheta2);

    glm::vec3 T = getDifferentDir(normal);
    T = glm::normalize(glm::cross(T, normal));
    glm::vec3 B = glm::normalize(glm::cross(T, normal));

    glm::vec3 h{ T * glm::cos(phi) * sinTheta + B * glm::sin(phi) * sinTheta + normal * cosTheta };
    h = glm::normalize(h);

    float denom = ((a2 - 1) * cosTheta2 + 1);
    *pdf = (a2) * INV_PI / (denom * denom * 4 * fmaxf(glm::dot(-wi, h), 0.f) + FLT_EPSILON);

    return glm::normalize(glm::reflect(wi, h));
}

__device__ __host__ glm::vec3 cosHemisphereSampler(const glm::vec3& normal, float* pdf, float rnd1, float rnd2) {
    float phi = rnd1 * TWO_PI;
    float r = sqrtf(rnd2);

    glm::vec3 wo_tan{ glm::cos(phi) * r, glm::sin(phi) * r, sqrtf(1.0f - rnd2) };

    *pdf = wo_tan.z * INV_PI;

    glm::vec3 T = getDifferentDir(normal);
    T = glm::normalize(glm::cross(T, normal));
    glm::vec3 B = glm::normalize(glm::cross(T, normal));

    return glm::normalize(T * wo_tan.x + B * wo_tan.y + normal * wo_tan.z);
}

__device__ __host__ glm::vec3 refractSampler(float ior, const glm::vec3& wi, glm::vec3 n, float rnd) {
    float invEta, iorI, iorT;
    float cosI = glm::dot(wi, n);
    if (cosI < 0) { // enter
        cosI = -cosI;
        n = -n;
        invEta = 1.f / ior;
        iorI = 1.f;
        iorT = ior;
    }
    else { // leave
        n = n;
        invEta = ior;
        iorI = ior;
        iorT = 1.f;
    }

    float cosT = 1.f - invEta * invEta * fmaxf(0.f, (1.f - cosI * cosI));
    if (cosT <= 0.f) { // internal reflection
        return glm::reflect(wi, -n);
    }
    else {
        cosT = sqrtf(cosT);
        float F = fresnel(cosI, cosT, iorI, iorT);

        if (rnd < F) {
            return glm::reflect(wi, -n);
        }
        return wi * invEta + n * (cosT - invEta * cosI);
    }
}

__device__ __host__ glm::vec3 reflectSampler(
    float metallic, glm::vec3 albedo, const glm::vec3& wi, glm::vec3 n, float rnd1, float rnd2, float rnd3, float* pdf, bool* specular) {
    float hv = glm::dot(n, -wi);
    float F = metallic + (1.f-metallic) * powf(1.f - fmaxf(hv, 0.f), 5.f);

    if (rnd1 > F) { // diffuse
        *specular = false;
        return cosHemisphereSampler(n, pdf, rnd2, rnd3);
    }
    else { // specular
        *specular = true;
        *pdf = 1.f;
        return glm::reflect(wi, n);
    }
}

__device__ __host__ glm::vec3 uniformHemisphereSampler(const glm::vec3& normal, float* pdf, float rnd1, float rnd2) {
    float phi = rnd1 * TWO_PI;
    float r = sqrtf(1.0f - rnd2 * rnd2);

    glm::vec3 wo_tan{ glm::cos(phi) * r, glm::sin(phi) * r, rnd2 };

    *pdf = INV_TWO_PI;

    glm::vec3 T = getDifferentDir(normal);
    T = glm::normalize(glm::cross(T, normal));
    glm::vec3 B = glm::normalize(glm::cross(T, normal));

    return T * wo_tan.x + B * wo_tan.y + normal * wo_tan.z;
}

}
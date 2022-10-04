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
__device__ __host__ glm::vec3 nagi::GGXImportanceSampler(float alpha, const glm::vec3& wi, const glm::vec3& normal, float* pdf, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);
    float rnd1 = u01(rng);
    float rnd2 = u01(rng);

    float a2 = alpha * alpha;
    float phi = rnd1 * TWO_PI;
    float cosTheta = sqrtf((1 - rnd2) / (rnd2 * (a2 - 1) + 1));
    float cosTheta2 = cosTheta * cosTheta;
    float sinTheta = sqrtf(1 - cosTheta2);

    glm::vec3 T = getDifferentDir(normal);
    T = glm::normalize(glm::cross(T, normal));
    glm::vec3 B = glm::normalize(glm::cross(T, normal));

    glm::vec3 h{ T * cosf(phi) * sinTheta + B * sinf(phi) * sinTheta + normal * cosTheta };

    float denom = ((a2 - 1) * cosTheta2 + 1);
    *pdf = (a2 * cosTheta * sinTheta) * INV_PI / (denom * denom * fabsf(glm::dot(wi, h)) + FLT_EPSILON);

    return glm::normalize(glm::reflect(wi, h));
}

__device__ __host__ glm::vec3 cosHemisphereSampler(const glm::vec3& normal, float* pdf, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float rnd1 = u01(rng);
    float rnd2 = u01(rng);

    float phi = rnd1 * TWO_PI;
    float r = sqrtf(rnd2);

    glm::vec3 wo_tan{ cosf(phi) * r, sinf(phi) * r, sqrtf(1.0f - rnd2) };

    *pdf = wo_tan.z * INV_PI;

    glm::vec3 T = getDifferentDir(normal);
    T = glm::normalize(glm::cross(T, normal));
    glm::vec3 B = glm::normalize(glm::cross(T, normal));

    return glm::normalize(T * wo_tan.x + B * wo_tan.y + normal * wo_tan.z);
}

__device__ __host__ glm::vec3 uniformHemisphereSampler(const glm::vec3& normal, float* pdf, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float rnd1 = u01(rng);
    float rnd2 = u01(rng);

    float phi = rnd1 * TWO_PI;
    float r = sqrtf(1.0f - rnd2 * rnd2);

    glm::vec3 wo_tan{ cosf(phi) * r, sinf(phi) * r, rnd2 };

    *pdf = INV_TWO_PI;

    glm::vec3 T = getDifferentDir(normal);
    T = glm::normalize(glm::cross(T, normal));
    glm::vec3 B = glm::normalize(glm::cross(T, normal));

    return T * wo_tan.x + B * wo_tan.y + normal * wo_tan.z;
}

}
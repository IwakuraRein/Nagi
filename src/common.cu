#include "common.cuh"

#include <glm/gtx/matrix_decompose.hpp>

bool doesFileExist(const std::string& name) {
	if (FILE* file = fopen(name.c_str(), "r")) {
		fclose(file);
		return true;
	}
	else {
		return false;
	}
}

std::string getFileName(const std::string& str) {
	int i = str.size() - 1;
	for (; i != 0; i--) {
		if (str[i] == '/')
			break;
	}
	if (str[i] == '/') i++;
	return str.substr(i, str.size() - i);
}

std::string strLeftStrip(const std::string& str, const std::string& strip) {
	if (strip.size() > str.size()) return str;
	int i = 0;
	for (; i < strip.size(); i++) {
		if (strip[i] != str[i]) return str;
	}
	std::string out;
	out.resize(str.size() - strip.size());
	for (; i < str.size(); i++) {
		out[i - strip.size()] = str[i];
	}
	return out;
}

std::string strRightStrip(const std::string& str, const std::string& strip) {
	if (strip.size() > str.size()) return str;
	for (int i = 0; i < strip.size(); i++) {
		if (strip[strip.size() - i] != str[str.size() - i]) return str;
	}
	std::string out;
	out.resize(str.size() - strip.size());
	for (int i = 0; i < out.size(); i++) {
		out[i] = str[i];
	}
	return out;
}


namespace nagi {

Scene scene;

__device__ __host__ void vecTransform(glm::vec3* vec, const glm::mat4& mat, float T) {
	glm::vec4 tmp{ mat * glm::vec4{ *vec, T } };
	if (glm::epsilonNotEqual(T, 0.f, FLT_EPSILON)) {
		if (glm::epsilonNotEqual(tmp.w, 0.f, FLT_EPSILON)) {
			vec->x = tmp.x / tmp.w;
			vec->y = tmp.y / tmp.w;
			vec->z = tmp.z / tmp.w;
		}
		else {
			*vec = glm::vec3{ 0.f };
		}
	}
	else *vec = tmp;
}

__device__ __host__ void vecTransform2(glm::vec3* vec, const glm::mat4& mat, float T) {
	*vec = mat * glm::vec4{ *vec, T };
}

__device__ __host__ void updateTransformMat(Transform* t) {
	t->transformMat = getTransformMat(t->position, t->rotation, t->scale);
	t->invTransformMat = glm::inverse(t->transformMat);
}

__device__ __host__ glm::mat4 getTransformMat(const glm::vec3& position, const glm::vec3& rotation, const glm::vec3& scale) {
	glm::mat4 mat = getRotationMat(rotation);
	mat[0] *= scale.x;
	mat[1] *= scale.y;
	mat[2] *= scale.z;
	mat[3] = glm::vec4{ position.x, position.y, position.z, 1.f };
	return mat;
}

__device__ __host__ glm::mat4 getRotationMat(const glm::vec3& rotation) {
	//const float c3 = glm::cos(glm::radians(rotation.z));
	//const float s3 = glm::sin(glm::radians(rotation.z));
	//const float c2 = glm::cos(glm::radians(rotation.x));
	//const float s2 = glm::sin(glm::radians(rotation.x));
	//const float c1 = glm::cos(glm::radians(rotation.y));
	//const float s1 = glm::sin(glm::radians(rotation.y));

	//return glm::mat4{
	//	{
	//		(c1 * c3 + s1 * s2 * s3),
	//		(c2 * s3),
	//		(c1 * s2 * s3 - c3 * s1),
	//		0.f
	//	},
	//	{
	//		(c3 * s1 * s2 - c1 * s3),
	//		(c2 * c3),
	//		(c1 * c3 * s2 + s1 * s3),
	//		0.f
	//	},
	//	{
	//		(c2 * s1),
	//		(-s2),
	//		(c1 * c2),
	//		0.f
	//	},
	//	{ 0.f, 0.f, 0.f, 1.f }
	//};

	return glm::yawPitchRoll(rotation.y, rotation.x, rotation.z);
}

//__device__ __host__ void updateCameraMat(Camera& cam) {
//	cam.projectMat = glm::perspective(cam.fov, cam.aspect, cam.near, cam.far);
//	cam.invProjectMat = glm::inverse(cam.projectMat);
//	cam.invViewMat = getTransformMat(cam.position, cam.rotation, { 1.f, 1.f, 1.f });
//	cam.viewMat = glm::inverse(cam.invViewMat);
//
//	glm::mat4 rot = getRotationMat(cam.rotation);
//	cam.forwardDir = glm::vec3{ 0.f, 0.f, 1.f };
//	cam.upDir = glm::vec3{ 0.f, -1.f, 0.f };
//	cam.rightDir = glm::vec3{ 1.f, 0.f, 0.f };
//	vecTransform(&cam.forwardDir, rot);
//	vecTransform(&cam.upDir, rot);
//	vecTransform(&cam.rightDir, rot);
//}

//__device__ __host__ void moveCamera(Camera& cam, const glm::vec3& vec) {
//	cam.position += vec;
//	updateCameraMat(cam);
//}

//__device__ __host__ void orientateCamera(Camera& cam, const glm::vec3& dir, const glm::vec3& up) {
//	cam.forwardDir = glm::normalize(dir);
//	cam.upDir = glm::normalize(up);
//	cam.rightDir = glm::cross(dir, up);
//
//	cam.viewMat = glm::lookAt(cam.position, cam.position + dir, cam.upDir);
//	cam.invViewMat = glm::inverse(cam.viewMat);
//
//	glm::vec3 scale, translation, skew;
//	glm::vec4 perspective;
//	glm::quat orientation;
//	glm::decompose(cam.invViewMat, scale, orientation, translation, skew, perspective);
//	cam.rotation.y = glm::yaw(orientation);
//	cam.rotation.x = glm::pitch(orientation);
//	cam.rotation.z = glm::roll(orientation);
//}

}

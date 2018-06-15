#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <memory>
#include <random>
#include <omp.h>


template <class T>
T clamp(T x, T xmin, T xmax) {
    if(x < xmin) return xmin;
    else if(x > xmax) return xmax;
    else return x;
}


struct Vec3 {
    float x;
    float y;
    float z;

    Vec3() { x = y = z = 0; };
    Vec3(float x) : x(x), y(x), z(x) {};
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {};

    Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    };

    float length() const {
        return std::sqrt(x*x + y*y + z*z);
    };
    float length2() const {
        return x*x + y*y + z*z;
    };
};
inline Vec3 operator+(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
inline Vec3 operator+(const Vec3& v, float k) {
    return Vec3(v.x + k, v.y + k, v.z + k);
}
inline Vec3 operator+(float k, const Vec3& v) {
    return v + k;
}

inline Vec3 operator-(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
inline Vec3 operator-(const Vec3& v, float k) {
    return Vec3(v.x - k, v.y - k, v.z - k);
}
inline Vec3 operator-(float k, const Vec3& v) {
    return Vec3(k - v.x, k - v.y, k - v.z);
}

inline Vec3 operator*(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}
inline Vec3 operator*(const Vec3& v, float k) {
    return Vec3(v.x * k, v.y * k, v.z * k);
}
inline Vec3 operator*(float k, const Vec3& v) {
    return Vec3(k * v.x, k * v.y, k * v.z);
}

inline Vec3 operator/(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}
inline Vec3 operator/(const Vec3& v, float k) {
    return Vec3(v.x / k, v.y / k, v.z / k);
}
inline Vec3 operator/(float k, const Vec3& v) {
    return Vec3(k / v.x, k / v.y, k / v.z);
}

inline std::ostream& operator<<(std::ostream& stream, const Vec3& v) {
    stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return stream;
}

inline float dot(const Vec3& v1, const Vec3& v2) {
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}
inline Vec3 cross(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
}

inline Vec3 normalize(const Vec3& v) {
    return v/v.length();
}

inline Vec3 pow(const Vec3& v, float n) {
    return Vec3(std::pow(v.x, n), std::pow(v.y, n), std::pow(v.z, n));
}

inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2*dot(v, n)*n;
}


struct Ray {
    Vec3 origin;
    Vec3 direction;

    Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {};

    Vec3 operator()(float t) const {
        return origin + t*direction;
    };
};


struct Image {
    int width;
    int height;
    Vec3* data;

    Image(int width, int height) : width(width), height(height) {
        data = new Vec3[width*height];
    };
    ~Image() {
        delete[] data;
    };

    Vec3 getPixel(int i, int j) const {
        if(i < 0 || i >= width || j < 0 || j >= height) {
            std::cerr << "Invalid Access" << std::endl;
            std::exit(1);
        }
        return data[j + width*i];
    };

    void setPixel(int i, int j, const Vec3& col) {
        if(i < 0 || i >= width || j < 0 || j >= height) {
            std::cerr << "Invalid Access" << std::endl;
            std::exit(1);
        }
        data[j + width*i] = col;
    };

    void divide(float k) {
        for(int i = 0; i < width; i++) {
            for(int j = 0; j < height; j++) {
                this->setPixel(i, j, this->getPixel(i, j)/k);
            }
        }
    };

    void gamma_correction() {
        for(int i = 0; i < width; i++) {
            for(int j = 0; j < height; j++) {
                Vec3 col = pow(this->getPixel(i, j), 1.0f/2.2f);
                this->setPixel(i, j, col);
            }
        }
    };

    void ppm_output(const std::string& filename) const {
        std::ofstream file(filename);

        file << "P3" << std::endl;
        file << width << " " << height << std::endl;
        file << 255 << std::endl;

        for(int j = 0; j < height; j++) {
            for(int i = 0; i < width; i++) {
                Vec3 col = this->getPixel(i, j);
                int r = 255*clamp(col.x, 0.0f, 1.0f);
                int g = 255*clamp(col.y, 0.0f, 1.0f);
                int b = 255*clamp(col.z, 0.0f, 1.0f);
                file << r << " " << g << " " << b << std::endl;
            }
        }

        file.close();
    };
};


struct Sphere;
struct Hit {
    float t;
    Vec3 hitPos;
    Vec3 hitNormal;
    const Sphere* hitSphere;

    Hit() {
        t = 1000000;
        hitSphere = nullptr;
    };
};


struct Sphere {
    Vec3 center;
    float radius;
    std::string type;
    Vec3 color;

    Sphere(const Vec3& center, float radius, const std::string& type, const Vec3& color) : center(center), radius(radius), type(type), color(color) {};

    bool intersect(const Ray& ray, Hit& res) const {
        float a = ray.direction.length2();
        float b = 2*dot(ray.direction, ray.origin - center);
        float c = (ray.origin - center).length2() - radius*radius;
        float D = b*b - 4*a*c;
        if(D < 0) return false;

        float t0 = (-b - std::sqrt(D))/(2*a);
        float t1 = (-b + std::sqrt(D))/(2*a);

        float t = t0;
        if(t < 0) {
            t = t1;
            if(t < 0) return false;
        }

        res.t = t;
        res.hitPos = ray(t);
        res.hitNormal = normalize(res.hitPos - center);
        res.hitSphere = this;

        return true;
    };
};


struct Camera {
    Vec3 camPos;
    Vec3 camForward;
    Vec3 camRight;
    Vec3 camUp;

    Camera(const Vec3& camPos, const Vec3& camForward) : camPos(camPos), camForward(camForward) {
        camRight = -normalize(cross(camForward, Vec3(0, 1, 0)));
        camUp = normalize(cross(camRight, camForward));
    };

    Ray getRay(float u, float v) const {
        return Ray(camPos, normalize(camForward + u*camRight + v*camUp));
    };
};


struct Accel {
    std::vector<std::shared_ptr<Sphere>> spheres;

    Accel() {};

    void add(const std::shared_ptr<Sphere>& p) {
        spheres.push_back(p);
    };

    bool intersect(const Ray& ray, Hit& res) const {
        bool isHit = false;
        for(auto sphere : spheres) {
            Hit res_each;
            if(sphere->intersect(ray, res_each)) {
                isHit = true;
                if(res_each.t < res.t || !isHit) {
                    res = res_each;
                }
            }
        }
        return isHit;
    };
};


std::random_device rnd_dev;
std::mt19937 mt(rnd_dev());
std::uniform_real_distribution<> dist(0, 1);
inline float rnd() {
    return dist(mt);
}


inline void orthonormalBasis(const Vec3& n, Vec3& x, Vec3& z) {
    if(n.x > 0.9) x = Vec3(0, 1, 0);
    else x = Vec3(1, 0, 0);
    x = x - dot(x, n)*n;
    x = normalize(x);
    z = normalize(cross(n, x));
}
inline Vec3 randomHemisphere(float& pdf, const Vec3& n) {
    pdf = 1/(2*M_PI);
    float u = rnd();
    float v = rnd();

    float x = std::cos(2*M_PI*u)*std::sqrt(1 - v*v);
    float y = v;
    float z = std::sin(2*M_PI*u)*std::sqrt(1 - v*v);
    Vec3 xv, zv;
    orthonormalBasis(n, xv, zv);
    return x*xv + y*n + z*zv;
}
inline Vec3 randomCosineHemisphere(float &pdf, const Vec3& n) {
    float u = rnd();
    float v = rnd();

    float theta = 0.5*std::acos(1 - 2*u);
    float phi = 2*M_PI*v;
    pdf = 1/M_PI * std::cos(theta);

    float x = std::cos(phi)*std::sin(theta);
    float y = std::cos(theta);
    float z = std::sin(phi)*std::sin(theta);
    Vec3 xv, zv;
    orthonormalBasis(n, xv, zv);
    return x*xv + y*n + z*zv;
}


Accel accel;


Vec3 getRadiance(const Ray& ray, int depth = 0) {
    if(depth == 10) return Vec3(0, 0, 0);

    Hit res;
    if(accel.intersect(ray, res)) {
        if(res.hitSphere->type == "diffuse") {
            float pdf;
            Vec3 nextDir = randomCosineHemisphere(pdf, res.hitNormal);
            Ray nextRay(res.hitPos + 0.01f*res.hitNormal, nextDir);
            float cos_term = std::max(dot(nextDir, res.hitNormal), 0.0f);
            return 1/pdf * res.hitSphere->color/M_PI * cos_term * getRadiance(nextRay, depth + 1);
        }
        else if(res.hitSphere->type == "mirror") {
            Vec3 nextDir = reflect(ray.direction, res.hitNormal);
            Ray nextRay(res.hitPos + 0.01f*res.hitNormal, nextDir);
            return res.hitSphere->color * getRadiance(nextRay, depth + 1);
        }
        else if(res.hitSphere->type == "light") {
            return res.hitSphere->color;
        }
        else {
            return Vec3(0, 0, 0);
        }
    }
    else {
        return Vec3(0, 0, 0);
    }
}


inline std::string percentage(float x, float max) {
    return std::to_string(x/max*100) + "%";
}
inline std::string progressbar(float x, float max) {
    const int max_count = 40;
    int cur_count = (int)(x/max * max_count);
    std::string str;
    str += "[";
    for(int i = 0; i < cur_count; i++)
        str += "#";
    for(int i = 0; i < (max_count - cur_count - 1); i++)
        str += " ";
    str += "]";
    return str;
}


int main() {
    const int samples = 1000;
    Image img(512, 512);
    Camera cam(Vec3(0, 1, 0), Vec3(0, 0, 1));

    //Walls
    accel.add(std::make_shared<Sphere>(Vec3(0, -10000, 0), 10000, "diffuse", Vec3(0.8)));
    accel.add(std::make_shared<Sphere>(Vec3(0, 10003, 0), 10000, "diffuse", Vec3(0.8)));
    accel.add(std::make_shared<Sphere>(Vec3(10001.5, 0, 0), 10000, "diffuse", Vec3(0, 0.2, 0.8)));
    accel.add(std::make_shared<Sphere>(Vec3(-10001.5, 0, 0), 10000, "diffuse", Vec3(0.8, 0.2, 0)));
    accel.add(std::make_shared<Sphere>(Vec3(0, 0, 10005), 10000, "diffuse", Vec3(0.8)));
    
    //Light
    accel.add(std::make_shared<Sphere>(Vec3(0, 3, 2.5), 0.5, "light", Vec3(10)));

    //Spheres
    accel.add(std::make_shared<Sphere>(Vec3(-0.7, 0.5, 3.0), 0.5, "diffuse", Vec3(0.8)));
    accel.add(std::make_shared<Sphere>(Vec3(0.7, 0.5, 2.5), 0.5, "diffuse", Vec3(0.8)));

#pragma omp parallel for schedule(dynamic, 1)
    for(int k = 0; k < samples; k++) {
        for(int i = 0; i < img.width; i++) {
            for(int j = 0; j < img.height; j++) {
                float u = (2.0*(i + rnd()) - img.width)/img.width;
                float v = (2.0*(j + rnd()) - img.height)/img.height;
                Ray ray = cam.getRay(u, v);
                img.setPixel(i, j, img.getPixel(i, j) + getRadiance(ray));
            }
        }
        if(omp_get_thread_num() == 0) {
            std::cout << progressbar(k, samples) << " " << percentage(k, samples) << "\r" << std::flush;
        }
    }
    img.divide(samples);
    img.gamma_correction();
    img.ppm_output("output.ppm");
    return 0;
}

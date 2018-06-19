#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <memory>
#include <random>
#include <omp.h>
#include <unistd.h>


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
    void operator+=(const Vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
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
    return normalize(v - 2.0f*dot(v, n)*n);
}
inline float fresnel(const Vec3& v, const Vec3& n, float n1, float n2) {
    float f0 = std::pow((n1 - n2)/(n1 + n2), 2.0f);
    return f0 + (1.0f - f0)*std::pow(1.0f - dot(v, n), 5.0f);
}
inline bool refract(const Vec3& v, const Vec3& n, float n1, float n2, Vec3& r) {
    float eta = n1/n2;
    float eta2 = eta*eta;
    float cosI = std::max(dot(-v, n), 0.0f);
    float sin2I = std::max(1.0f - cosI*cosI, 0.0f);
    if(sin2I >= 1) return false;
    float cosT = std::sqrt(1.0f - eta2*sin2I);
    r = normalize(eta*v + (eta*cosI - cosT)*n);
    return true;
}


struct Ray {
    Vec3 origin;
    Vec3 direction;

    Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {};

    Vec3 operator()(float t) const {
        return origin + t*direction;
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
inline Vec3 randomSphere(float &pdf) {
    pdf = 1/(4*M_PI);
    float u = rnd();
    float v = rnd();

    float y = 1 - 2*v;
    float x = std::cos(2*M_PI*u)*std::sqrt(std::max(1 - y*y, 0.0f));
    float z = std::sin(2*M_PI*u)*std::sqrt(std::max(1 - y*y, 0.0f));
    return Vec3(x, y, z);
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
    bool inside;

    Hit() {
        t = 10000000;
        hitSphere = nullptr;
        inside = false;
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
        if(t < 0.005f) {
            t = t1;
            if(t < 0.005f) return false;
        }

        res.t = t;
        res.hitPos = ray(t);
        res.hitNormal = normalize(res.hitPos - center);
        res.hitSphere = this;
        res.inside = dot(ray.direction, res.hitNormal) > 0 ? true : false;

        return true;
    };

    Vec3 samplePos(float& pdf, Vec3& normal) const {
        Vec3 samplePos = center + radius*randomSphere(pdf);
        normal = normalize(samplePos - center);
        pdf = 1/(4*M_PI*radius*radius);
        return samplePos;
    };

    Vec3 samplePos2(const Vec3& dir, float& pdf, Vec3& normal) const {
        Vec3 samplePos = center + radius*randomHemisphere(pdf, -dir);
        normal = normalize(samplePos - center);
        pdf = 1/(2*M_PI*radius*radius);
        return samplePos;
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
                if(res_each.t < res.t) {
                    isHit = true;
                    res = res_each;
                }
            }
        }
        return isHit;
    };
};


struct Light {
    std::vector<std::shared_ptr<Sphere>> lights;

    Light() {};

    void add(const std::shared_ptr<Sphere>& p) {
        lights.push_back(p);
    };
};


Accel accel;
Light light;


const float eps = 0.0f;
Vec3 getRadiance(const Ray& ray, int depth = 0, float roulette = 1.0f) {
    if(depth > 10) {
        roulette *= 0.9f;
    }
    if(rnd() >= roulette) {
        return Vec3(0, 0, 0);
    }

    Hit res;
    if(accel.intersect(ray, res)) {
        if(res.hitSphere->type == "diffuse") {
            Vec3 color;
            //light sampling
            for(auto l : light.lights) {
                float lightPdf;
                Vec3 lightNormal;
                Vec3 lightCenterDir = normalize(l->center - res.hitPos);
                //Vec3 lightPos = l->samplePos(lightPdf, lightNormal);
                Vec3 lightPos = l->samplePos2(lightCenterDir, lightPdf, lightNormal);
                Vec3 lightDir = normalize(lightPos - res.hitPos);

                float dot1 = dot(res.hitNormal, lightDir);
                float dot2 = dot(-lightDir, lightNormal);
                if(dot1 < 0 || dot2 < 0) {
                    continue;
                }

                Ray shadowRay(res.hitPos + eps*res.hitNormal, lightDir);
                Hit hit_shadow;
                if(!accel.intersect(shadowRay, hit_shadow)) {
                    std::cout << "shadowRay hit nothing, origin:" << shadowRay.origin << ", direction:" << shadowRay.direction << std::endl;
                    continue;
                }

                if(hit_shadow.hitSphere == &(*l) && (lightPos - hit_shadow.hitPos).length() < 0.001f) {
                    float dist2 = (lightPos - res.hitPos).length2();
                    float geometry_term = dot1 * 1/dist2 * dot2;
                    color += 1/roulette * 1/lightPdf * geometry_term * l->color * res.hitSphere->color/M_PI;
                }
            }

            float dirPdf;
            Vec3 nextDir = randomCosineHemisphere(dirPdf, res.hitNormal);
            Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
            float cos_term = std::max(dot(nextDir, res.hitNormal), 0.0f);
            return color + 1/roulette * 1/(dirPdf + 0.001f) * res.hitSphere->color/M_PI * cos_term * getRadiance(nextRay, depth + 1, roulette);
        }
        else if(res.hitSphere->type == "light") {
            if(depth == 0) {
                return res.hitSphere->color;
            }
            else {
                return Vec3(0, 0, 0);
            }
        }
        else {
            return Vec3(0, 0, 0);
        }
        /*
        else if(res.hitSphere->type == "mirror") {
            Vec3 nextDir = reflect(ray.direction, res.hitNormal);
            Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
            return 1/roulette * res.hitSphere->color * getRadiance(nextRay, depth + 1, roulette);
        }
        else if(res.hitSphere->type == "glass") {
            if(!res.inside) {
                float fr = fresnel(-ray.direction, res.hitNormal, 1.0f, 1.4f);
                //reflect
                if(rnd() < fr) {
                    Vec3 nextDir = reflect(ray.direction, res.hitNormal);
                    Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
                    return 1/roulette * res.hitSphere->color * getRadiance(nextRay, depth + 1, roulette);
                }
                else {
                    Vec3 nextDir;
                    if(refract(ray.direction, res.hitNormal, 1.0f, 1.4f, nextDir)) {
                        Ray nextRay(res.hitPos - eps*res.hitNormal, nextDir);
                        return 1/roulette * std::pow(1.4f/1.0f, 2.0f) * res.hitSphere->color * getRadiance(nextRay, depth + 1, roulette);
                    }
                    else {
                        std::cerr << "Something Wrong!!" << std::endl;
                        return Vec3(0, 0, 0);
                    }
                }
            }
            else {
                float fr = fresnel(-ray.direction, -res.hitNormal, 1.4f, 1.0f);
                //reflect
                if(rnd() < fr) {
                    Vec3 nextDir = reflect(ray.direction, -res.hitNormal);
                    Ray nextRay(res.hitPos - eps*res.hitNormal, nextDir);
                    return 1/roulette * res.hitSphere->color * getRadiance(nextRay, depth + 1, roulette);
                }
                //refract
                else {
                    Vec3 nextDir;
                    if(refract(ray.direction, -res.hitNormal, 1.4f, 1.0f, nextDir)) {
                        Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
                        return 1/roulette * std::pow(1.0f/1.4f, 2.0f) * res.hitSphere->color * getRadiance(nextRay, depth + 1, roulette);
                    }
                    //total reflection
                    else {
                        nextDir = reflect(ray.direction, -res.hitNormal);
                        Ray nextRay(res.hitPos - eps*res.hitNormal, nextDir);
                        return 1/roulette * res.hitSphere->color * getRadiance(nextRay, depth + 1, roulette);
                    }
                }
            }
        }
        else if(res.hitSphere->type == "light") {
            return res.hitSphere->color;
        }
        else {
            return Vec3(0, 0, 0);
        }
        */
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


int main(int argc, char** argv) {
    int width;
    int height;
    int samples;

    int opt;
    while((opt = getopt(argc, argv, "w:h:s:")) != -1) {
        switch(opt) {
            case 'w':
                width = std::stoi(optarg);
                break;
            case 'h':
                height = std::stoi(optarg);
                break;
            case 's':
                samples = std::stoi(optarg);
                break;
        }
    }

    Image img(width, height);
    Camera cam(Vec3(0, 1, 0), Vec3(0, 0, 1));

    //Walls
    accel.add(std::make_shared<Sphere>(Vec3(0, -10000, 0), 10000, "diffuse", Vec3(0.8)));
    accel.add(std::make_shared<Sphere>(Vec3(0, 10003, 0), 10000, "diffuse", Vec3(0.8)));
    accel.add(std::make_shared<Sphere>(Vec3(10001.5, 0, 0), 10000, "diffuse", Vec3(0.25, 0.5, 1.0)));
    accel.add(std::make_shared<Sphere>(Vec3(-10001.5, 0, 0), 10000, "diffuse", Vec3(1.0, 0.3, 0.3)));
    accel.add(std::make_shared<Sphere>(Vec3(0, 0, 10005), 10000, "diffuse", Vec3(0.8)));
    
    //Light
    auto p = std::make_shared<Sphere>(Vec3(0, 2.0, 2.5), 0.1, "light", Vec3(30));
    accel.add(p);
    light.add(p);

    //Spheres
    auto sphere1 = std::make_shared<Sphere>(Vec3(-0.7, 0.5, 3.0), 0.5, "diffuse", Vec3(0.8));
    auto sphere2 = std::make_shared<Sphere>(Vec3(0.7, 0.5, 2.5), 0.5, "diffuse", Vec3(0.8));
    accel.add(sphere1);
    accel.add(sphere2);

    for(int k = 0; k < samples; k++) {
        for(int i = 0; i < img.width; i++) {
#pragma omp parallel for schedule(dynamic, 1)
            for(int j = 0; j < img.height; j++) {
                float u = (2.0*(i + rnd()) - img.width)/img.width;
                float v = (2.0*(j + rnd()) - img.height)/img.height;
                Ray ray = cam.getRay(u, v);
                Vec3 color = getRadiance(ray);
                if(std::isnan(color.x) || std::isnan(color.y) || std::isnan(color.z)) {
                    std::cout << "nan detected" << std::endl;
                    color = Vec3(0, 0, 0);
                }
                if(color.x < 0 || color.y < 0 || color.z < 0) {
                    std::cout << "minus detected" << std::endl;
                    color = Vec3(0, 0, 0);
                }
                img.setPixel(i, j, img.getPixel(i, j) + color);
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

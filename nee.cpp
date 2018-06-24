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


typedef float Real;


template <class T>
T clamp(T x, T xmin, T xmax) {
    if(x < xmin) return xmin;
    else if(x > xmax) return xmax;
    else return x;
}


struct Vec3 {
    Real x;
    Real y;
    Real z;

    Vec3() { x = y = z = 0; };
    Vec3(Real x) : x(x), y(x), z(x) {};
    Vec3(Real x, Real y, Real z) : x(x), y(y), z(z) {};

    Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    };
    void operator+=(const Vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
    };

    Real length() const {
        return std::sqrt(x*x + y*y + z*z);
    };
    Real length2() const {
        return x*x + y*y + z*z;
    };
};
inline Vec3 operator+(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}
inline Vec3 operator+(const Vec3& v, Real k) {
    return Vec3(v.x + k, v.y + k, v.z + k);
}
inline Vec3 operator+(Real k, const Vec3& v) {
    return v + k;
}

inline Vec3 operator-(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}
inline Vec3 operator-(const Vec3& v, Real k) {
    return Vec3(v.x - k, v.y - k, v.z - k);
}
inline Vec3 operator-(Real k, const Vec3& v) {
    return Vec3(k - v.x, k - v.y, k - v.z);
}

inline Vec3 operator*(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}
inline Vec3 operator*(const Vec3& v, Real k) {
    return Vec3(v.x * k, v.y * k, v.z * k);
}
inline Vec3 operator*(Real k, const Vec3& v) {
    return Vec3(k * v.x, k * v.y, k * v.z);
}

inline Vec3 operator/(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}
inline Vec3 operator/(const Vec3& v, Real k) {
    return Vec3(v.x / k, v.y / k, v.z / k);
}
inline Vec3 operator/(Real k, const Vec3& v) {
    return Vec3(k / v.x, k / v.y, k / v.z);
}

inline std::ostream& operator<<(std::ostream& stream, const Vec3& v) {
    stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return stream;
}

inline Real dot(const Vec3& v1, const Vec3& v2) {
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}
inline Vec3 cross(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
}

inline Vec3 normalize(const Vec3& v) {
    return v/v.length();
}

inline Vec3 pow(const Vec3& v, Real n) {
    return Vec3(std::pow(v.x, n), std::pow(v.y, n), std::pow(v.z, n));
}

inline Vec3 reflect(const Vec3& v, const Vec3& n) {
    return normalize(v - 2.0*dot(v, n)*n);
}
inline Real fresnel(const Vec3& v, const Vec3& n, Real n1, Real n2) {
    Real f0 = std::pow((n1 - n2)/(n1 + n2), 2.0);
    return f0 + (1.0 - f0)*std::pow(1.0 - dot(v, n), 5.0);
}
inline bool refract(const Vec3& v, const Vec3& n, Real n1, Real n2, Vec3& r) {
    Real eta = n1/n2;
    Real eta2 = eta*eta;
    Real cosI = std::max(dot(-v, n), (Real)0.0);
    Real sin2I = std::max((Real)1.0 - cosI*cosI, (Real)0.0);
    if(sin2I >= 1) return false;
    Real cosT = std::sqrt((Real)1.0 - eta2*sin2I);
    r = normalize(eta*v + (eta*cosI - cosT)*n);
    return true;
}


struct Ray {
    Vec3 origin;
    Vec3 direction;

    Ray(const Vec3& origin, const Vec3& direction) : origin(origin), direction(direction) {};

    Vec3 operator()(Real t) const {
        return origin + t*direction;
    };
};


std::random_device rnd_dev;
std::mt19937 mt(rnd_dev());
std::uniform_real_distribution<> dist(0, 1);
inline Real rnd() {
    return dist(mt);
}


inline void orthonormalBasis(const Vec3& n, Vec3& x, Vec3& z) {
    if(n.x > 0.9) x = Vec3(0, 1, 0);
    else x = Vec3(1, 0, 0);
    x = x - dot(x, n)*n;
    x = normalize(x);
    z = normalize(cross(n, x));
}
inline Vec3 randomSphere(Real &pdf) {
    pdf = 1/(4*M_PI);
    Real u = rnd();
    Real v = rnd();

    Real y = 1 - 2*v;
    Real x = std::cos(2*M_PI*u)*std::sqrt(std::max(1 - y*y, (Real)0.0));
    Real z = std::sin(2*M_PI*u)*std::sqrt(std::max(1 - y*y, (Real)0.0));
    return Vec3(x, y, z);
}
inline Vec3 randomHemisphere(Real& pdf, const Vec3& n) {
    pdf = 1/(2*M_PI);
    Real u = rnd();
    Real v = rnd();

    Real x = std::cos(2*M_PI*u)*std::sqrt(1 - v*v);
    Real y = v;
    Real z = std::sin(2*M_PI*u)*std::sqrt(1 - v*v);
    Vec3 xv, zv;
    orthonormalBasis(n, xv, zv);
    return x*xv + y*n + z*zv;
}
inline Vec3 randomCosineHemisphere(Real &pdf, const Vec3& n) {
    Real u = rnd();
    Real v = rnd();

    Real theta = 0.5*std::acos(1 - 2*u);
    Real phi = 2*M_PI*v;
    pdf = 1/M_PI * std::cos(theta);

    Real x = std::cos(phi)*std::sin(theta);
    Real y = std::cos(theta);
    Real z = std::sin(phi)*std::sin(theta);
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

    void divide(Real k) {
        for(int i = 0; i < width; i++) {
            for(int j = 0; j < height; j++) {
                this->setPixel(i, j, this->getPixel(i, j)/k);
            }
        }
    };

    void gamma_correction() {
        for(int i = 0; i < width; i++) {
            for(int j = 0; j < height; j++) {
                Vec3 col = pow(this->getPixel(i, j), 1.0/2.2);
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
                int r = 255*clamp(col.x, (Real)0.0, (Real)1.0);
                int g = 255*clamp(col.y, (Real)0.0, (Real)1.0);
                int b = 255*clamp(col.z, (Real)0.0, (Real)1.0);
                file << r << " " << g << " " << b << std::endl;
            }
        }

        file.close();
    };
};


struct Sphere;
struct Hit {
    Real t;
    Vec3 hitPos;
    Vec3 hitNormal;
    const Sphere* hitSphere;
    bool inside;

    Hit() {
        t = 10000;
        hitSphere = nullptr;
        inside = false;
    };
};


struct Sphere {
    Vec3 center;
    Real radius;
    std::string type;
    Vec3 color;

    Sphere(const Vec3& center, Real radius, const std::string& type, const Vec3& color) : center(center), radius(radius), type(type), color(color) {};

    bool intersect(const Ray& ray, Hit& res) const {
        Real a = ray.direction.length2();
        Real b = 2.0*dot(ray.direction, ray.origin - center);
        Real c = (ray.origin - center).length2() - radius*radius;
        Real D = b*b - 4.0*a*c;
        if(D < 0) return false;

        Real t0 = (-b - std::sqrt(D))/(2*a);
        Real t1 = (-b + std::sqrt(D))/(2*a);

        Real t = t0;
        if(t < 0.005) {
            t = t1;
            if(t < 0.005) return false;
        }

        res.t = t;
        res.hitPos = ray(t);
        res.hitNormal = normalize(res.hitPos - center);
        res.hitSphere = this;
        res.inside = dot(ray.direction, res.hitNormal) > 0 ? true : false;

        return true;
    };

    Vec3 samplePos(Real& pdf, Vec3& normal) const {
        Vec3 samplePos = center + radius*randomSphere(pdf);
        normal = normalize(samplePos - center);
        pdf = 1/(4*M_PI*radius*radius);
        return samplePos;
    };

    Vec3 samplePos2(const Vec3& dir, Real& pdf, Vec3& normal) const {
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

    Ray getRay(Real u, Real v) const {
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


const Real eps = 0;
Vec3 getRadiance(const Ray& ray, int depth = 0, Real roulette = 1.0, bool nee_flag = false) {
    if(depth > 10) {
        roulette *= 0.9;
    }
    if(rnd() >= roulette) {
        return Vec3(0, 0, 0);
    }

    Vec3 color;
    Hit res;
    if(accel.intersect(ray, res)) {
        if(!nee_flag) {
            if(res.hitSphere->type == "light") {
                color += res.hitSphere->color;
            }
        }

        if(res.hitSphere->type == "diffuse") {
            //light sampling
            for(auto l : light.lights) {
                Real lightPdf;
                Vec3 lightNormal;
                Vec3 lightCenterDir = normalize(l->center - res.hitPos);
                //Vec3 lightPos = l->samplePos(lightPdf, lightNormal);
                Vec3 lightPos = l->samplePos2(lightCenterDir, lightPdf, lightNormal);
                Vec3 lightDir = normalize(lightPos - res.hitPos);

                Real dot1 = dot(res.hitNormal, lightDir);
                Real dot2 = dot(-lightDir, lightNormal);
                if(dot1 < 0 || dot2 < 0) {
                    continue;
                }

                Ray shadowRay(res.hitPos + eps*res.hitNormal, lightDir);
                Hit hit_shadow;
                if(!accel.intersect(shadowRay, hit_shadow)) {
                    std::cerr << "shadowRay doesn't hit anything, origin:" << shadowRay.origin << ", direction:" << shadowRay.direction << std::endl;
                    continue;
                }

                if(hit_shadow.hitSphere == &(*l) && (lightPos - hit_shadow.hitPos).length() < 0.001) {
                    Real dist2 = (lightPos - res.hitPos).length2();
                    Real geometry_term = dot1 * 1/dist2 * dot2;
                    color += 1/roulette * 1/lightPdf * geometry_term * l->color * res.hitSphere->color/M_PI;
                }
            }

            Real dirPdf;
            Vec3 nextDir = randomCosineHemisphere(dirPdf, res.hitNormal);
            Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
            Real cos_term = std::max(dot(nextDir, res.hitNormal), (Real)0.0);
            color += 1/roulette * 1/(dirPdf + 0.001) * res.hitSphere->color/M_PI * cos_term * getRadiance(nextRay, depth + 1, roulette, true);
        }
        else if(res.hitSphere->type == "light") {
            Real dirPdf;
            Vec3 nextDir = randomCosineHemisphere(dirPdf, res.hitNormal);
            Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
            Real cos_term = std::max(dot(nextDir, res.hitNormal), (Real)0.0);
            color += 1/roulette * 1/(dirPdf + 0.001) * 1/M_PI * cos_term * getRadiance(nextRay, depth + 1, roulette, true);
        }
        else if(res.hitSphere->type == "mirror") {
            Vec3 nextDir = reflect(ray.direction, res.hitNormal);
            Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
            color += 1/roulette * res.hitSphere->color * getRadiance(nextRay, depth + 1, roulette, false);
        }
        else if(res.hitSphere->type == "glass") {
            if(!res.inside) {
                Real fr = fresnel(-ray.direction, res.hitNormal, 1.0, 1.4);
                //reflect
                if(rnd() < fr) {
                    Vec3 nextDir = reflect(ray.direction, res.hitNormal);
                    Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
                    return 1/roulette * res.hitSphere->color * getRadiance(nextRay, depth + 1, roulette, false);
                }
                else {
                    Vec3 nextDir;
                    if(refract(ray.direction, res.hitNormal, 1.0, 1.4, nextDir)) {
                        Ray nextRay(res.hitPos - eps*res.hitNormal, nextDir);
                        return 1/roulette * std::pow(1.4/1.0, 2.0) * res.hitSphere->color * getRadiance(nextRay, depth + 1, roulette, false);
                    }
                    else {
                        std::cerr << "Something Wrong!!" << std::endl;
                        return Vec3(0, 0, 0);
                    }
                }
            }
            else {
                Real fr = fresnel(-ray.direction, -res.hitNormal, 1.4, 1.0);
                //reflect
                if(rnd() < fr) {
                    Vec3 nextDir = reflect(ray.direction, -res.hitNormal);
                    Ray nextRay(res.hitPos - eps*res.hitNormal, nextDir);
                    return 1/roulette * res.hitSphere->color * getRadiance(nextRay, depth + 1, roulette, false);
                }
                //refract
                else {
                    Vec3 nextDir;
                    if(refract(ray.direction, -res.hitNormal, 1.4, 1.0, nextDir)) {
                        Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
                        return 1/roulette * std::pow(1.0/1.4, 2.0) * res.hitSphere->color * getRadiance(nextRay, depth + 1, roulette, false);
                    }
                    //total reflection
                    else {
                        nextDir = reflect(ray.direction, -res.hitNormal);
                        Ray nextRay(res.hitPos - eps*res.hitNormal, nextDir);
                        return 1/roulette * res.hitSphere->color * getRadiance(nextRay, depth + 1, roulette, false);
                    }
                }
            }
        }
        else {
            color = Vec3(0, 0, 0);
        }
    }
    else {
        color = Vec3(0, 0, 0);
    }
    return color;
}


inline std::string percentage(Real x, Real max) {
    return std::to_string(x/max*100) + "%";
}
inline std::string progressbar(Real x, Real max) {
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
    accel.add(std::make_shared<Sphere>(Vec3(0, 0, -10005), 10000, "diffuse", Vec3(0.8)));
    
    //Light
    auto p = std::make_shared<Sphere>(Vec3(0, 2.5, 2.5), 0.2, "light", Vec3(30));
    accel.add(p);
    light.add(p);

    //Spheres
    auto sphere1 = std::make_shared<Sphere>(Vec3(-0.7, 0.5, 3.0), 0.5, "mirror", Vec3(1.0));
    auto sphere2 = std::make_shared<Sphere>(Vec3(0.7, 0.5, 2.5), 0.5, "glass", Vec3(1.0));
    accel.add(sphere1);
    accel.add(sphere2);

    for(int k = 0; k < samples; k++) {
        for(int i = 0; i < img.width; i++) {
#pragma omp parallel for schedule(dynamic, 1)
            for(int j = 0; j < img.height; j++) {
                Real u = (2.0*(i + rnd()) - img.width)/img.width;
                Real v = (2.0*(j + rnd()) - img.height)/img.height;
                Ray ray = cam.getRay(u, v);
                Vec3 color = getRadiance(ray);
                if(std::isnan(color.x) || std::isnan(color.y) || std::isnan(color.z)) {
                    std::cerr << "nan detected" << std::endl;
                    color = Vec3(0, 0, 0);
                }
                if(std::isinf(color.x) || std::isinf(color.y) || std::isinf(color.z)) {
                    std::cerr << "inf detected" << std::endl;
                    color = Vec3(0, 0, 0);
                }
                if(color.x < 0 || color.y < 0 || color.z < 0) {
                    std::cerr << "minus detected" << std::endl;
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

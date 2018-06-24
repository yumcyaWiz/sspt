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

                float rf = clamp(col.x, 0.0f, 1.0f);
                float gf = clamp(col.y, 0.0f, 1.0f);
                float bf = clamp(col.z, 0.0f, 1.0f);

                int r = 255*rf;
                int g = 255*gf;
                int b = 255*bf;
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
        t = 1000000;
        hitSphere = nullptr;
        inside = false;
    };
};


typedef float Spectrum;


struct Sphere {
    Vec3 center;
    float radius;
    std::string type;
    Spectrum color;

    Sphere(const Vec3& center, float radius, const std::string& type, const Spectrum& color) : center(center), radius(radius), type(type), color(color) {};

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
        res.inside = dot(ray.direction, res.hitNormal) > 0 ? true : false;

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
                if(res_each.t < res.t) {
                    isHit = true;
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


inline float SCHOTT_BK7(float l) {
    return std::sqrt(1 + 1.03961212*l*l/(l*l - 0.00600069867) + 0.231792344*l*l/(l*l - 0.0200179144) + 1.01046945*l*l/(l*l - 103.560653));
}
inline float SCHOTT_F(float l) {
    float v = std::sqrt(1 + 1.34533359*l*l/(l*l - 0.00997743871) + 0.209073176*l*l/(l*l - 0.0470450767) + 0.937357162*l*l/(l*l - 111.886764));
    return v*v*v/(1.5*1.5*1.5);
}


static const float wavelength_to_xyz[] = {
	1.222E-07,1.3398E-08,5.35027E-07, 
	9.1927E-07,1.0065E-07,4.0283E-06, 
	5.9586E-06,6.511E-07,2.61437E-05, 
	0.000033266,0.000003625,0.00014622, 
	0.000159952,0.000017364,0.000704776, 
	0.00066244,0.00007156,0.0029278, 
	0.0023616,0.0002534,0.0104822, 
	0.0072423,0.0007685,0.032344, 
	0.0191097,0.0020044,0.0860109, 
	0.0434,0.004509,0.19712, 
	0.084736,0.008756,0.389366, 
	0.140638,0.014456,0.65676, 
	0.204492,0.021391,0.972542, 
	0.264737,0.029497,1.2825, 
	0.314679,0.038676,1.55348, 
	0.357719,0.049602,1.7985, 
	0.383734,0.062077,1.96728, 
	0.386726,0.074704,2.0273, 
	0.370702,0.089456,1.9948, 
	0.342957,0.106256,1.9007, 
	0.302273,0.128201,1.74537, 
	0.254085,0.152761,1.5549, 
	0.195618,0.18519,1.31756, 
	0.132349,0.21994,1.0302, 
	0.080507,0.253589,0.772125, 
	0.041072,0.297665,0.57006, 
	0.016172,0.339133,0.415254, 
	0.005132,0.395379,0.302356, 
	0.003816,0.460777,0.218502, 
	0.015444,0.53136,0.159249, 
	0.037465,0.606741,0.112044, 
	0.071358,0.68566,0.082248, 
	0.117749,0.761757,0.060709, 
	0.172953,0.82333,0.04305, 
	0.236491,0.875211,0.030451, 
	0.304213,0.92381,0.020584, 
	0.376772,0.961988,0.013676, 
	0.451584,0.9822,0.007918, 
	0.529826,0.991761,0.003988, 
	0.616053,0.99911,0.001091, 
	0.705224,0.99734,0, 
	0.793832,0.98238,0, 
	0.878655,0.955552,0, 
	0.951162,0.915175,0, 
	1.01416,0.868934,0, 
	1.0743,0.825623,0, 
	1.11852,0.777405,0, 
	1.1343,0.720353,0, 
	1.12399,0.658341,0, 
	1.0891,0.593878,0, 
	1.03048,0.527963,0, 
	0.95074,0.461834,0, 
	0.856297,0.398057,0, 
	0.75493,0.339554,0, 
	0.647467,0.283493,0, 
	0.53511,0.228254,0, 
	0.431567,0.179828,0, 
	0.34369,0.140211,0, 
	0.268329,0.107633,0, 
	0.2043,0.081187,0, 
	0.152568,0.060281,0, 
	0.11221,0.044096,0, 
	0.0812606,0.0318004,0, 
	0.05793,0.0226017,0, 
	0.0408508,0.0159051,0, 
	0.028623,0.0111303,0, 
	0.0199413,0.0077488,0, 
	0.013842,0.0053751,0, 
	0.00957688,0.00371774,0, 
	0.0066052,0.00256456,0, 
	0.00455263,0.00176847,0, 
	0.0031447,0.00122239,0, 
	0.00217496,0.00084619,0, 
	0.0015057,0.00058644,0, 
	0.00104476,0.00040741,0, 
	0.00072745,0.000284041,0, 
	0.000508258,0.00019873,0, 
	0.00035638,0.00013955,0, 
	0.000250969,0.000098428,0, 
	0.00017773,0.000069819,0, 
	0.00012639,0.000049737,0, 
	0.000090151,3.55405E-05,0, 
	6.45258E-05,0.000025486,0, 
	0.000046339,1.83384E-05,0, 
	3.34117E-05,0.000013249,0, 
	0.000024209,9.6196E-06,0, 
	1.76115E-05,7.0128E-06,0, 
	0.000012855,5.1298E-06,0, 
	9.41363E-06,3.76473E-06,0, 
	0.000006913,2.77081E-06,0, 
	5.09347E-06,2.04613E-06,0, 
	3.7671E-06,1.51677E-06,0, 
	2.79531E-06,1.12809E-06,0, 
	0.000002082,8.4216E-07,0, 
	1.55314E-06,6.297E-07,0, 
};


inline Vec3 xyz_to_rgb(const Vec3& xyz) {
    return Vec3(dot(Vec3(2.3655, -0.8971, -0.4683), xyz), dot(Vec3(-0.5151, 1.4264, 0.0887), xyz), dot(Vec3(0.0052, -0.0144, 1.0089), xyz));
}


Accel accel;


const float eps = 0.005f;
Spectrum getRadiance(const Ray& ray, double wave_length, int depth = 0, float roulette = 1.0f) {
    if(depth > 10) {
        roulette *= 0.9f;
    }
    if(rnd() >= roulette) {
        return 0;
    }

    Hit res;
    if(accel.intersect(ray, res)) {
        if(res.hitSphere->type == "diffuse") {
            float pdf;
            Vec3 nextDir = randomCosineHemisphere(pdf, res.hitNormal);
            Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
            float cos_term = std::max(dot(nextDir, res.hitNormal), 0.0f);
            return 1/roulette * 1/pdf * res.hitSphere->color/M_PI * cos_term * getRadiance(nextRay, wave_length, depth + 1, roulette);
        }
        else if(res.hitSphere->type == "mirror") {
            Vec3 nextDir = reflect(ray.direction, res.hitNormal);
            Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
            return 1/roulette * res.hitSphere->color * getRadiance(nextRay, wave_length, depth + 1, roulette);
        }
        else if(res.hitSphere->type == "glass") {
            if(!res.inside) {
                float n1 = 1.0f;
                float n2 = SCHOTT_F(wave_length);
                float fr = fresnel(-ray.direction, res.hitNormal, n1, n2);
                //reflect
                if(rnd() < fr) {
                    Vec3 nextDir = reflect(ray.direction, res.hitNormal);
                    Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
                    return 1/roulette * res.hitSphere->color * getRadiance(nextRay, wave_length, depth + 1, roulette);
                }
                else {
                    Vec3 nextDir;
                    if(refract(ray.direction, res.hitNormal, n1, n2, nextDir)) {
                        Ray nextRay(res.hitPos - eps*res.hitNormal, nextDir);
                        return 1/roulette * std::pow(n2/n1, 2.0f) * res.hitSphere->color * getRadiance(nextRay, wave_length, depth + 1, roulette);
                    }
                    else {
                        std::cerr << "Something Wrong!!" << std::endl;
                        return 0;
                    }
                }
            }
            else {
                float n1 = SCHOTT_F(wave_length);
                float n2 = 1.0f;
                float fr = fresnel(-ray.direction, -res.hitNormal, n1, n2);
                //reflect
                if(rnd() < fr) {
                    Vec3 nextDir = reflect(ray.direction, -res.hitNormal);
                    Ray nextRay(res.hitPos - eps*res.hitNormal, nextDir);
                    return 1/roulette * res.hitSphere->color * getRadiance(nextRay, wave_length, depth + 1, roulette);
                }
                //refract
                else {
                    Vec3 nextDir;
                    if(refract(ray.direction, -res.hitNormal, n1, n2, nextDir)) {
                        Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
                        return 1/roulette * std::pow(n2/n1, 2.0f) * res.hitSphere->color * getRadiance(nextRay, wave_length, depth + 1, roulette);
                    }
                    //total reflection
                    else {
                        nextDir = reflect(ray.direction, -res.hitNormal);
                        Ray nextRay(res.hitPos - eps*res.hitNormal, nextDir);
                        return 1/roulette * res.hitSphere->color * getRadiance(nextRay, wave_length, depth + 1, roulette);
                    }
                }
            }
        }
        else if(res.hitSphere->type == "light") {
            return res.hitSphere->color;
        }
        else {
            return 0;
        }
    }
    else {
        return 0.2*std::pow(dot(ray.direction, normalize(Vec3(1, 1, 1))), 32.0);
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
    Camera cam(Vec3(0, 30, -90), Vec3(0, 0, 1));

    //Walls
    accel.add(std::make_shared<Sphere>(Vec3(0, -10000, 0), 10000, "diffuse", 0.8));
    //accel.add(std::make_shared<Sphere>(Vec3(0, 10003, 0), 10000, "diffuse", 0.8));
    //accel.add(std::make_shared<Sphere>(Vec3(10001.5, 0, 0), 10000, "diffuse", 0.8));
    //accel.add(std::make_shared<Sphere>(Vec3(-10001.5, 0, 0), 10000, "diffuse", 0.8));
    //accel.add(std::make_shared<Sphere>(Vec3(0, 0, 10005), 10000, "diffuse", 0.8));
    
    //Light
    //accel.add(std::make_shared<Sphere>(Vec3(0, 3.0, 2.5), 0.5, "light", 0.1));

    //Spheres
    accel.add(std::make_shared<Sphere>(Vec3(-0.7 - 30, 30, 3.0), 30, "diffuse", 1.0));
    accel.add(std::make_shared<Sphere>(Vec3(0.7 + 30, 30, 2.5), 30, "glass", 1.0));

    //波長の分割数
    const int wl_count = 95;
    float wl_pdf[wl_count];
    float wl_cdf[wl_count];

    //Y成分＝輝度成分
    float wl_luminance[wl_count];
    for(int i = 0; i < wl_count; i++) {
        wl_luminance[i] = wavelength_to_xyz[3*i + 1];
    }

    //波長ごとの輝度のc.d.fとp.d.fを作る
    float luminance_sum = 0;
    for(int i = 0; i < wl_count; i++) {
        luminance_sum += wl_luminance[i];
        wl_cdf[i] = luminance_sum;
        wl_pdf[i] = wl_luminance[i];
    }
    for(int i = 0; i < wl_count; i++) {
        wl_cdf[i] /= luminance_sum;
        wl_pdf[i] /= luminance_sum;
    }


    for(int k = 0; k < samples; k++) {
        for(int i = 0; i < img.width; i++) {
#pragma omp parallel for schedule(dynamic, 1)
            for(int j = 0; j < img.height; j++) {
                float u = (2.0*(i + rnd()) - img.width)/img.width;
                float v = (2.0*(j + rnd()) - img.height)/img.height;
                Ray ray = cam.getRay(u, v);

                //波長の重点的サンプリング
                int wl_index = std::lower_bound(wl_cdf, wl_cdf + wl_count, rnd()) - wl_cdf;
                if(wl_index >= wl_count) wl_index = wl_count - 1;
                float wave_length_pdf = wl_pdf[wl_index];
                //int wl_index = (int)(wl_count*rnd());
                float wave_length = (wl_index * 5 + 360)/1000.0;

                Spectrum spec = getRadiance(ray, wave_length)/wave_length_pdf;
                Vec3 xyz = Vec3(wavelength_to_xyz[3*wl_index], wavelength_to_xyz[3*wl_index + 1], wavelength_to_xyz[3*wl_index + 2])*spec;
                Vec3 color = xyz_to_rgb(xyz);

                if(color.x < 0 || std::isnan(color.x)) color.x = 0;
                if(color.y < 0 || std::isnan(color.y)) color.y = 0;
                if(color.z < 0 || std::isnan(color.z)) color.z = 0;

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

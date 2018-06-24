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
        t = 1000000;
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


inline double SCHOTT_BK7(double l) {
    return std::sqrt(1 + 1.03961212*l*l/(l*l - 0.00600069867) + 0.231792344*l*l/(l*l - 0.0200179144) + 1.01046945*l*l/(l*l - 103.560653));
}
inline double SCHOTT_F(double l) {
    return std::sqrt(1 + 1.34533359*l*l/(l*l - 0.00997743871) + 0.209073176*l*l/(l*l - 0.0470450767) + 0.937357162*l*l/(l*l - 111.886764));
}


static const float wavelength_to_xyz[] = {
    2.952420E-03,4.076779E-04,1.318752E-02,
    7.641137E-03,1.078166E-03,3.424588E-02,
    1.879338E-02,2.589775E-03,8.508254E-02,
    4.204986E-02,5.474207E-03,1.927065E-01,
    8.277331E-02,1.041303E-02,3.832822E-01,
    1.395127E-01,1.712968E-02,6.568187E-01,
    2.077647E-01,2.576133E-02,9.933444E-01,
    2.688989E-01,3.529554E-02,1.308674E+00,
    3.281798E-01,4.698226E-02,1.624940E+00,
    3.693084E-01,6.047429E-02,1.867751E+00,
    4.026189E-01,7.468288E-02,2.075946E+00,
    4.042529E-01,8.820537E-02,2.132574E+00,
    3.932139E-01,1.039030E-01,2.128264E+00,
    3.482214E-01,1.195389E-01,1.946651E+00,
    3.013112E-01,1.414586E-01,1.768440E+00,
    2.534221E-01,1.701373E-01,1.582342E+00,
    1.914176E-01,1.999859E-01,1.310576E+00,
    1.283167E-01,2.312426E-01,1.010952E+00,
    7.593120E-02,2.682271E-01,7.516389E-01,
    3.836770E-02,3.109438E-01,5.549619E-01,
    1.400745E-02,3.554018E-01,3.978114E-01,
    3.446810E-03,4.148227E-01,2.905816E-01,
    5.652072E-03,4.780482E-01,2.078158E-01,
    1.561956E-02,5.491344E-01,1.394643E-01,
    3.778185E-02,6.248296E-01,8.852389E-02,
    7.538941E-02,7.012292E-01,5.824484E-02,
    1.201511E-01,7.788199E-01,3.784916E-02,
    1.756832E-01,8.376358E-01,2.431375E-02,
    2.380254E-01,8.829552E-01,1.539505E-02,
    3.046991E-01,9.233858E-01,9.753000E-03,
    3.841856E-01,9.665325E-01,6.083223E-03,
    4.633109E-01,9.886887E-01,3.769336E-03,
    5.374170E-01,9.907500E-01,2.323578E-03,
    6.230892E-01,9.997775E-01,1.426627E-03,
    7.123849E-01,9.944304E-01,8.779264E-04,
    8.016277E-01,9.848127E-01,5.408385E-04,
    8.933408E-01,9.640545E-01,3.342429E-04,
    9.721304E-01,9.286495E-01,2.076129E-04,
    1.034327E+00,8.775360E-01,1.298230E-04,
    1.106886E+00,8.370838E-01,8.183954E-05,
    1.147304E+00,7.869950E-01,5.207245E-05,
    1.160477E+00,7.272309E-01,3.347499E-05,
    1.148163E+00,6.629035E-01,2.175998E-05,
    1.113846E+00,5.970375E-01,1.431231E-05,
    1.048485E+00,5.282296E-01,9.530130E-06,
    9.617111E-01,4.601308E-01,6.426776E-06,
    8.629581E-01,3.950755E-01,0.000000E+00,
    7.603498E-01,3.351794E-01,0.000000E+00,
    6.413984E-01,2.751807E-01,0.000000E+00,
    5.290979E-01,2.219564E-01,0.000000E+00,
    4.323126E-01,1.776882E-01,0.000000E+00,
    3.496358E-01,1.410203E-01,0.000000E+00,
    2.714900E-01,1.083996E-01,0.000000E+00,
    2.056507E-01,8.137687E-02,0.000000E+00,
    1.538163E-01,6.033976E-02,0.000000E+00,
    1.136072E-01,4.425383E-02,0.000000E+00,
    8.281010E-02,3.211852E-02,0.000000E+00,
    5.954815E-02,2.302574E-02,0.000000E+00,
    4.221473E-02,1.628841E-02,0.000000E+00,
    2.948752E-02,1.136106E-02,0.000000E+00,
    2.025590E-02,7.797457E-03,0.000000E+00,
    1.410230E-02,5.425391E-03,0.000000E+00,
    9.816228E-03,3.776140E-03,0.000000E+00,
    6.809147E-03,2.619372E-03,0.000000E+00,
    4.666298E-03,1.795595E-03,0.000000E+00,
    3.194041E-03,1.229980E-03,0.000000E+00,
    2.205568E-03,8.499903E-04,0.000000E+00,
    1.524672E-03,5.881375E-04,0.000000E+00,
    1.061495E-03,4.098928E-04,0.000000E+00,
    7.400120E-04,2.860718E-04,0.000000E+00,
    5.153113E-04,1.994949E-04,0.000000E+00,
    3.631969E-04,1.408466E-04,0.000000E+00,
    2.556624E-04,9.931439E-05,0.000000E+00,
    1.809649E-04,7.041878E-05,0.000000E+00,
    1.287394E-04,5.018934E-05,0.000000E+00,
    9.172477E-05,3.582218E-05,0.000000E+00,
    6.577532E-05,2.573083E-05,0.000000E+00,
    4.708916E-05,1.845353E-05,0.000000E+00,
    3.407653E-05,1.337946E-05,0.000000E+00,
    2.469630E-05,9.715798E-06,0.000000E+00,
    1.794555E-05,7.074424E-06,0.000000E+00,
    1.306345E-05,5.160948E-06,0.000000E+00,
    9.565993E-06,3.788729E-06,0.000000E+00,
    7.037621E-06,2.794625E-06,0.000000E+00,
    5.166853E-06,2.057152E-06,0.000000E+00,
    3.815429E-06,1.523114E-06,0.000000E+00,
    2.837980E-06,1.135758E-06,0.000000E+00,
    2.113325E-06,8.476168E-07,0.000000E+00,
    1.579199E-06,6.345380E-07,0.000000E+00
};


inline Vec3 xyz_to_rgb(const Vec3& xyz) {
    return Vec3(dot(Vec3(2.3655, -0.8971, -0.4683), xyz), dot(Vec3(-0.5151, 1.4264, 0.0887), xyz), dot(Vec3(0.0052, -0.0144, 1.0089), xyz));
}


Accel accel;


const float eps = 0.005f;
Vec3 getRadiance(const Ray& ray, double wave_length, int depth = 0, float roulette = 1.0f) {
    if(depth > 10) {
        roulette *= 0.9f;
    }
    if(rnd() >= roulette) {
        return Vec3(0, 0, 0);
    }

    Hit res;
    if(accel.intersect(ray, res)) {
        if(res.hitSphere->type == "diffuse") {
            float pdf;
            Vec3 nextDir = randomCosineHemisphere(pdf, res.hitNormal);
            Ray nextRay(res.hitPos + eps*res.hitNormal, nextDir);
            float cos_term = std::max(dot(nextDir, res.hitNormal), 0.0f);
            return 1/roulette * 1/pdf * res.hitSphere->color/M_PI * cos_term * getRadiance(nextRay, depth + 1, roulette);
        }
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
    accel.add(std::make_shared<Sphere>(Vec3(0, 2.0, 2.5), 0.1, "light", Vec3(70)));

    //Spheres
    accel.add(std::make_shared<Sphere>(Vec3(-0.7, 0.5, 3.0), 0.5, "diffuse", Vec3(1.0)));
    accel.add(std::make_shared<Sphere>(Vec3(0.7, 0.5, 2.5), 0.5, "diffuse", Vec3(1.0)));

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

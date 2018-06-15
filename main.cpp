#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdlib>
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


struct Hit {
    float t;
    Vec3 hitPos;
    Vec3 hitNormal;

    Hit() {
        t = 1000000;
    };
};


struct Sphere {
    Vec3 center;
    float radius;

    Sphere(const Vec3& center, float radius) : center(center), radius(radius) {};

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



int main() {
    Image img(512, 512);
    Camera cam(Vec3(0, 0, -3), Vec3(0, 0, 1));
    Sphere sphere(Vec3(0, 0, 0), 1.0);
    for(int i = 0; i < img.width; i++) {
        for(int j = 0; j < img.height; j++) {
            float u = (2.0*i - img.width)/img.width;
            float v = (2.0*j - img.height)/img.height;
            Ray ray = cam.getRay(u, v);
            Hit res;
            if(sphere.intersect(ray, res)) {
                img.setPixel(i, j, (res.hitNormal + 1.0f)/2.0f);
            }
            else {
                img.setPixel(i, j, Vec3(0, 0, 0));
            }
        }
    }
    img.ppm_output("output.ppm");
    return 0;
}

#ifndef TENSOR_H
#define TENSOR_H
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include "tensortransform.h"

using namespace std;

class Tensor {
    friend Tensor dot(const Tensor &a , const Tensor &b);
    friend Tensor matmul(const Tensor &a , const Tensor &b);


    double* data;
    vector<size_t> shape;
    size_t total_size;


    Tensor(const vector<size_t>& shape) {
        data = nullptr;
        total_size = 1;
        if (shape.size() > 3) {
            cout << "El tensor no puede tener mas de 3 dimensiones." << endl;
            total_size = 0;
            return;
        }

        this->shape = shape;
        for (int i = 0; i < shape.size(); ++i) {
            total_size *= shape[i];
        }
        data = new double[total_size];
    }

public:

    Tensor(const vector<size_t>& shape, const vector<double>& values) {

        total_size = 1;

        if (shape.size() > 3) {
            cout << "El tensor no puede tener mas de 3 dimensiones." << endl;
            data = nullptr;
            total_size = 0;
            return;
        }

        for (int i = 0; i < shape.size(); ++i) {
            total_size *= shape[i];
        }

        if (values.size() != total_size) {
            cout<< "El numero de valores no coincide con el tamaño del tensor." << endl;
            data = nullptr;
            total_size = 0;
            return;
        }

        this->shape = shape;

        data = new double[total_size];
        for (size_t i = 0; i < total_size; ++i) {
            data[i] = values[i];
        }
    }


    Tensor(const Tensor& tensor) {
        shape = tensor.shape;
        total_size = tensor.total_size;

        if (tensor.data == nullptr) {
            data = nullptr;
        }
        else {
            data = new double[total_size];
            for (size_t i = 0; i < total_size; ++i) {
                data[i] = tensor.data[i];
            }
        }
    }


    Tensor(Tensor&& other) {
        data = other.data;
        shape = other.shape;
        total_size = other.total_size;

        other.data = nullptr;
        other.total_size = 0;
    }


    Tensor& operator=(const Tensor& other) {
        if (this != &other) {

            delete[] data;

            shape = other.shape;
            total_size = other.total_size;

            if (other.data != nullptr) {
                data = new double[total_size];
                for (size_t i = 0; i < total_size; ++i) {
                    data[i] = other.data[i];
                }
            } else {
                data = nullptr;
            }
        }
        return *this;
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {

            delete[] data;


            data = other.data;
            shape = std::move(other.shape);
            total_size = other.total_size;


            other.data = nullptr;
            other.total_size = 0;
        }
        return *this;
    }

    ~Tensor() {
        delete[] data;
    }

    vector<size_t> get_shape() const {;
        return shape;
    }


    static Tensor zeros(const vector<size_t>& shape) {
        Tensor tensor(shape);
        if (tensor.data) {
            for (int i = 0; i < tensor.total_size; ++i) {
                tensor.data[i] = 0.0;
            }
        }
        return tensor;

    }

    static Tensor ones(const vector<size_t>& shape) {
        Tensor tensor(shape);
        if (tensor.data) {
            for (int i = 0; i < tensor.total_size; ++i) {
                tensor.data[i] = 1.0;
            }
        }
        return tensor;
    }

    static Tensor random(const vector<size_t>& shape, double min, double max) {
        Tensor tensor(shape);
        if (tensor.data) {
            for (int i = 0; i < tensor.total_size; ++i) {
                double r = (double)rand() / RAND_MAX;
                tensor.data[i] = min + r * (max - min);
            }
        }
        return tensor;
        }

    static Tensor arange(double start, double end) {
        if (start >= end) {
            cout << "start debe ser menor a end" << endl;
            return Tensor({0}, {});
        }

        vector<double> values;
        for (double v = start; v < end; v += 1.0) {
            values.push_back(v);
        }

        return Tensor({values.size()}, values);
    }


    Tensor operator+(const Tensor& other) const {

        if (data == nullptr or other.data == nullptr) {
             cout << "Error: Intento de sumar tensores invalidos/vacios." << endl;
             return Tensor({0}, {});
        }


        if (shape != other.shape) {
            cout << "Error: Dimensiones incompatibles para la suma." << endl;
            return Tensor({0}, {});
        }

        Tensor res(shape);

        if (res.data == nullptr) {
            return res;
        }

        for (size_t i = 0; i < total_size; ++i) {
            res.data[i] = data[i] + other.data[i];
        }
        return res;
    }

    Tensor operator-(const Tensor& other) const {
        if (data == nullptr or other.data == nullptr)
            return Tensor({0}, {});
        if (shape != other.shape) {
            cout << "Error: Dimensiones incompatibles para la resta." << endl;
            return Tensor({0}, {});
        }

        Tensor res(shape);
        if (res.data == nullptr)
            return res;

        for (size_t i = 0; i < total_size; ++i) {
            res.data[i] = this->data[i] - other.data[i];
        }
        return res;
    }

    Tensor operator*(const Tensor& other) const {
        if (data == nullptr or other.data == nullptr)
            return Tensor({0}, {});
        if (shape != other.shape) {
            cout << "Error: Dimensiones incompatibles para multiplicacion." << endl;
            return Tensor({0}, {});
        }

        Tensor res(shape);
        if (res.data == nullptr) return res;

        for (size_t i = 0; i < total_size; ++i) {
            res.data[i] = this->data[i] * other.data[i];
        }
        return res;
    }

    Tensor operator*(double scalar) const {
        if (data == nullptr)
            return Tensor({0}, {});

        Tensor res(shape);
        if (res.data == nullptr)
            return res;

        for (size_t i = 0; i < total_size; ++i) {
            res.data[i] = data[i] * scalar;
        }
        return res;
    }


    Tensor view(const vector<size_t>& new_shape) const {
        if (data == nullptr)
            return Tensor({0}, {});

        int new_total = 1;
        for (size_t s : new_shape)
            new_total *= s;


        if (new_total != total_size) {
            cout << "Error El numero total de elementos no coincide." << endl;
            return *this;
        }

        if (new_shape.size() > 3) {
            cout << "Maximo 3 dimensiones permitidas." << endl;
            return *this;
        }

        Tensor res(*this);
        res.shape = new_shape;

        return res;
    }


    Tensor unsqueeze(size_t axis) const {
        if (shape.size() >= 3) {
            cout << "Error unsqueeze: Ya tiene 3 dimensiones." << endl;
            return *this;
        }

        vector<size_t> new_shape = shape;

        if (axis <= new_shape.size()) {
            new_shape.insert(new_shape.begin() + axis, 1);
        } else {
            new_shape.push_back(1);
        }

        return view(new_shape);
    }

static Tensor concat(const vector<Tensor>& tensors, size_t axis) {
        if (tensors.empty()) return Tensor({0}, {});

        vector<size_t> base_shape = tensors[0].shape;
        size_t final_dim = 0;

        for (const auto& t : tensors) {
            if (t.shape.size() != base_shape.size()) {
                cout << "Error: Distinto numero de dimensiones." << endl;
                return Tensor({0}, {});
            }
            for (size_t i = 0; i < base_shape.size(); ++i) {
                if (i != axis and t.shape[i] != base_shape[i]) {
                    cout << "Error: Dimensiones incompatibles." << endl;
                    return Tensor({0}, {});
                }
            }
            final_dim += t.shape[axis];
        }

        vector<size_t> res_shape = base_shape;
        res_shape[axis] = final_dim;
        Tensor result(res_shape);

        size_t repeats = 1;
        size_t block_size = 1;

        for (size_t i = 0; i < axis; ++i)
            repeats *= base_shape[i];
        for (size_t i = axis + 1; i < base_shape.size(); ++i)
            block_size *= base_shape[i];


        size_t write_index = 0;

        for (size_t r = 0; r < repeats; ++r) {
            for (const auto& t : tensors) {
                size_t elements_to_copy = t.shape[axis] * block_size;

                size_t read_index = r * elements_to_copy;

                for (size_t k = 0; k < elements_to_copy; ++k) {
                    result.data[write_index++] = t.data[read_index + k];
                }
            }
        }

        return result;
    }


    Tensor apply(const TensorTransform& t) const {
        if (data == nullptr)
            return Tensor({0}, {});

        Tensor res(shape);
        for (size_t i = 0; i < total_size; ++i) {
            res.data[i] = t.apply(data[i]);
        }
        return res;
    }
};

inline Tensor dot(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        cout << "Error: Los tensores deben tener las mismas dimensiones." << endl;
        return Tensor({0}, {});
    }

    double sum = 0.0;
    for (size_t i = 0; i < a.total_size; ++i) {
        sum += a.data[i] * b.data[i];
    }
    return Tensor({1}, {sum});
}

inline Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.shape.size() != 2 or b.shape.size() != 2 or a.shape[1] != b.shape[0]) {
        cout << "Error: Dimensiones invalidas." << endl;
        return Tensor::zeros({1, 1});
    }
    size_t M = a.shape[0], K = a.shape[1], N = b.shape[1];
    Tensor res = Tensor::zeros({M, N});

    for (size_t i = 0; i < M; i++) {
        for (size_t k = 0; k < K; k++) {
            double val_a = a.data[i * K + k];

            for (size_t j = 0; j < N; j++) {
                res.data[i * N + j] += val_a * b.data[k * N + j];
            }
        }
    }
    return res;
}

#endif

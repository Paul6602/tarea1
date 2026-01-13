#include <iostream>
#include <vector>
#include "Tensor.h"
// Asegúrate de que Tensor.h incluye "tensortransform.h"
// o inclúyelo aquí si es necesario: #include "tensortransform.h"

using namespace std;

int main() {

    Tensor input = Tensor::random({1000, 20, 20}, 0.0, 1.0);

    Tensor flattened = input.view({1000, 400});


    Tensor W1 = Tensor::random({400, 100}, -0.1, 0.1);
    Tensor layer1 = matmul(flattened, W1);


    Tensor b1 = Tensor::zeros({1000, 100});
    layer1 = layer1 + b1;

    ReLU relu;
    Tensor active1 = layer1.apply(relu);

    Tensor W2 = Tensor::random({100, 10}, -0.1, 0.1);
    Tensor layer2 = matmul(active1, W2);


    Tensor b2 = Tensor::zeros({1000, 10});
    layer2 = layer2 + b2;


    Sigmoid sigmoid;
    Tensor output = layer2.apply(sigmoid);


    vector<size_t> shape = output.get_shape();

    cout << "Dimension final: { ";
    for (size_t i = 0; i < shape.size(); ++i) {
        cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
    }
    cout << " }" << endl;

    return 0;
}
#include <vector>
#include <iostream>
#include <random>

class NeuralNetwork{

int num_inputs, num_hidden, num_output;

std::vector<double> inputs, hidden_biases, hidden_outputs, output_biases, outputs;
std::vector<std::vector<double>> input_hidden_weights, hidden_output_weights;

public:
   
   NeuralNetwork(int i, int h, int o)
   :num_inputs(i), num_hidden(h), num_output(o),
    inputs(i, 0), hidden_biases(h), hidden_outputs(h),
    input_hidden_weights(i, std::vector<double>(h,0.0)), 
    hidden_output_weights(h, std::vector<double>(o,0.0)),
    output_biases(o), outputs(o)
   {
      std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
      std::default_random_engine re;

      for(int j = 0; j < i; ++j)
        for(int m = 0; m < h; ++m)
         input_hidden_weights[j][m] = unif(re);

      for(j = 0; j < h; ++j)
         hidden_biases[j] = unif(re);

      for(j = 0; j < h; ++j)
        for(m = 0; m < o; ++m)
         hidden_output_weights[j][m] = unif(re);

      for(j = 0; j < o; ++j)
         output_biases[i] = unif(re);
   }

  std::vector<double>
  all_weights()
  {
      int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
      std::vector<double> result(numWeights);
      int k = 0;

      for (int i = 0; i < ihWeights.Length; ++i)
        for (int j = 0; j < ihWeights[0].Length; ++j, ++k)
          result[k] = ihWeights[i][j];

      for (int i = 0; i < hBiases.Length; ++i, ++k)
        result[k] = hBiases[i];

      for (int i = 0; i < hoWeights.Length; ++i)
        for (int j = 0; j < hoWeights[0].Length; ++j, ++k)
          result[k] = hoWeights[i][j];

      for (int i = 0; i < oBiases.Length; ++i, ++k)
        result[k] = oBiases[i];

      return result;
  }

  double
  HyperTan(double x)
  {
      if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
      else if (x > 20.0) return 1.0;
      else return Math.Tanh(x);
  }

};

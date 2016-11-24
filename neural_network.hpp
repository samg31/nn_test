#include <vector>
#include <iostream>
#include <random>
#include <cmath>

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
  hyper_tan(double x)
  {
      if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
      else if (x > 20.0) return 1.0;
      else return std::tanh(x);
  }

 std::vector<double>
 soft_max(std::vector<double> out_sum)
 {
   double max = out_sums[0];

   for(int i = 0; i < out_sums.size(); ++i)
     if(out_sums[i] > max) max = out_sums[i];

   double scale = 0;
   for(i = 0; i < out_sums.size(); ++i)
     scale += std::exp(out_sums[i] - max);

   std::vector<double> result(out_sums.size());
   for(i = 0; i < out_sums.size(); ++i)
     result[i] = std::exp(out_sums[i] - max) / scale;

   return result;
 } 

  int
  max_value_idx(std::vector<double> v)
  {
    int maxIdx = 0;
    double max = v[0];

    for(int i = 0; i < v.size(); ++i)
      if(v[i] > max)
       { 
         max = v[i];
         maxIdx = i;
       }

    return max;
  }



};

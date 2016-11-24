#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <cassert>

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

  void
  copy_weights(std::vector<double> new_values)
  {
    int num_weights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
    assert(num_weights == new_values.size());
    std::vector<double> result(numWeights);
    int k = 0;

      for (int i = 0; i < input_hidden_weights.Length; ++i)
        for (int j = 0; j < input_hidden_weights[0].Length; ++j, ++k)
          result[k] = input_hidden_weights[i][j];

      for (int i = 0; i < hidden_biases.Length; ++i, ++k)
        result[k] = hidden_biases[i];

      for (int i = 0; i < hidden_output_weights.Length; ++i)
        for (int j = 0; j < hidden_output_weights[0].Length; ++j, ++k)
          result[k] = hidden_output_weights[i][j];

      for (int i = 0; i < output_biases.Length; ++i, ++k)
        result[k] = output_biases[i];

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

  std::vector<double>
  compute_values(std::vector<double> in_vals) 
  {
    std::vector<double> hid_sums(num_hidden, 0), out_sums(num_output, 0);

    for(int i = 0; i < num_hidden; ++i)
       for(int j = 0; j < num_inputs; ++j)
          hid_sums[i] = in_vals[j] * input_hidden_weights[j][i];

    for(i = 0; i < num_hidden; ++i)
       hid_sums[i] += hidden_biases[i];
    
    for(i = 0; i < num_hidden; ++i)
       hidden_outputs[i] = hyper_tan(hid_sums[i]);
    
    for(i = 0; i < num_output; ++i)
       for(j = 0; j < num_hidden; ++j)
          out_sums[i] += hidden_outputs[j] * hidden_output_weights[j][i];
    
    for(i = 0; i < num_output; ++i)
       out_sums[i] += output_biases[i];
    
    outputs = soft_max(out_sums);

    return outputs;
      
  }

  double
  accuracy(std::vector<std::vector<double>> test_data, std::vector<double> weights)
  {
    copy_weights(weights);

    int right = 0;
    int wrong = 0;
    std::vector<double> in_vals(num_inputs), tar_vals(num_output), comp_vals;

    for(int i = 0; i < test_data.size(); ++i)
    {
      for(int j = 0; j < num_inputs; ++j)
          in_vals[j] = data[i][j];

      for(j = 0; j < num_output; ++j)
          target_vals[j] = data[i][j];

      comp_vals = compute_values(in_vals); 
      int maxIdx = max_value_idx(comp_vals);

      if(tar_vals[maxIdx] == 1)
        ++right;
      else
        ++wrong; 
    }
    return (right * 1.0) / (right + wrong);
  }

  double 
  sq_mean_err(std::vector<std::vector<double>> data, std::vector<double> weights)
  {
     std::vector<double> input_vals(num_inputs), target_vals(num_output);
     double err = 0;

     for(int i = 0; i < data.size(); ++i)
     {
        for(int j = 0; j < num_inputs; ++j)
          input_vals[j] = data[i][j];

        for(j = 0; j < num_output; ++j)
          target_vals[j] = data[i][j];

        std::vector<double> new_values = compute_values(input_vals);

        for(i = 0; i < new_values.size(); ++i)
           err += std::pow((new_values[i] - target_vals[i]), 2);
     }

     return err / data.size();
  }

};

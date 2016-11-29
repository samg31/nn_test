#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <iostream>
#include <random>
#include <cmath>
#include <cassert>
#include <omp.h>

class NeuralNetwork {
    
    int num_inputs, num_hidden, num_output;
    
    std::vector<double> inputs, hidden_biases, hidden_outputs, output_biases, outputs;
    std::vector<std::vector<double>> input_hidden_weights, hidden_output_weights;
    
public:
    
    NeuralNetwork(int i, int h, int o)
        :num_inputs(i), num_hidden(h), num_output(o),
          inputs(i, 0), hidden_biases(h), hidden_outputs(h),
          input_hidden_weights(i, std::vector<double>(h, 0.0)),
          hidden_output_weights(h, std::vector<double>(o, 0.0)),
          output_biases(o), outputs(o)
    {
        double lower_bound = 0.0001, upper_bound = 0.01;
        std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
        std::default_random_engine re;
#pragma omp parallel num_threads(num_hidden)
        {
#pragma omp for collapse(2)
            for (int j = 0; j < i; ++j)
                for (int m = 0; m < h; ++m)
                    input_hidden_weights[j][m] = unif(re);
#pragma omp for
            for (int j = 0; j < h; ++j)
                hidden_biases[j] = unif(re);
#pragma omp for collapse(2)
            for (int j = 0; j < h; ++j)
                for (int m = 0; m < o; ++m)
                    hidden_output_weights[j][m] = unif(re);
#pragma omp for
            for (int j = 0; j < o; ++j)
                output_biases[i] = unif(re);
        }
    }
    
    int
    sign(double x)
    {
        if(x > 0)
            return 1;
        else if (x < 0)
            return -1;
        else
            return 0;
    }
    
    std::vector<double>
    all_weights()
    {
        int numWeights = (num_inputs * num_hidden) + (num_hidden * num_output) + num_hidden + num_output;
        std::vector<double> result(numWeights);
        int offset = (num_inputs * num_hidden);
#pragma omp parallel num_threads(num_hidden)
        {
#pragma omp for collapse(2)
            for (int i = 0; i < input_hidden_weights.size(); ++i)
                for (int j = 0; j < input_hidden_weights[i].size(); ++j)
                    result[(i * input_hidden_weights.size()) + j] = input_hidden_weights[i][j];
#pragma omp for
            for (int i = 0; i < hidden_biases.size(); ++i)
                result[(i * hidden_biases.size())+offset] = hidden_biases[i];
            
            offset += num_hidden;
            
#pragma omp for collapse(2)
            for (int i = 0; i < hidden_output_weights.size(); ++i)
                for (int j = 0; j < hidden_output_weights[0].size(); ++j)
                    result[(i * hidden_output_weights.size()) + j + offset] = hidden_output_weights[i][j];
            
            offset += (num_hidden * num_output);
            
#pragma omp for
            for (int i = 0; i < output_biases.size(); ++i)
                result[(i * output_biases.size()) + offset] = output_biases[i];
        }
        return result;
        
    }
    
    double
    hyper_tan(double x)
    {
        if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
        else if (x > 20.0) return 1.0;
        else{
            return std::tanh(x);
        }
    }
    
    std::vector<double>
    soft_max(std::vector<double> out_sums)
    {
        double max = out_sums[0];
        
        for (int i = 0; i < out_sums.size(); ++i)
            if (out_sums[i] > max) max = out_sums[i];
        
        double scale = 0;
        
        for (int i = 0; i < out_sums.size(); ++i)
            scale += std::exp(out_sums[i] - max);
        
        std::vector<double> result(out_sums.size());
        for (int i = 0; i < out_sums.size(); ++i)
            result[i] = std::exp(out_sums[i] - max) / scale;
        return result;
    }
    
    int
    max_value_idx(std::vector<double> v)
    {
        int maxIdx = 0;
        double max = v[0];
        
        for (int i = 1; i < v.size(); ++i)
            if (v[i] > max)
            {
                max = v[i];
                maxIdx = i;
            }
        
        return maxIdx;
    }
    
    std::vector<double>
    compute_values(std::vector<double> in_vals)
    {
        std::vector<double> hid_sums(num_hidden, 0), out_sums(num_output, 0);
        
        for (int i = 0; i < num_hidden; ++i)
            for (int j = 0; j < num_inputs; ++j)
                hid_sums[i] += in_vals[j] * input_hidden_weights[j][i];
        
        for (int i = 0; i < num_hidden; ++i)
            hid_sums[i] += hidden_biases[i];
        
        for (int i = 0; i < num_hidden; ++i)
            hidden_outputs[i] = hyper_tan(hid_sums[i]);
        
        for (int i = 0; i < num_output; ++i)
            for (int j = 0; j < num_hidden; ++j)
                out_sums[i] += hidden_outputs[j] * hidden_output_weights[j][i];
        
        for (int i = 0; i < num_output; ++i)
            out_sums[i] += output_biases[i];
        
        outputs = soft_max(out_sums);
        
        return outputs;
        
    }
    
    double
    accuracy(std::vector<std::vector<double>> &test_data, std::vector<double> weights)
    {
        int right = 0;
        int wrong = 0;
        std::vector<double> in_vals(num_inputs), tar_vals(num_output), comp_vals;
        
        for (int i = 0; i < test_data.size(); ++i)
        {
            for (int j = 0; j < num_inputs; ++j)
                in_vals[j] = test_data[i][j];
            
            for (int j = 0; j < num_output; ++j)
                tar_vals[j] = test_data[i][j];
            
            comp_vals = compute_values(in_vals);
            int maxIdx = max_value_idx(comp_vals);
            
            if (tar_vals[maxIdx] == 1)
                ++right;
            else
                ++wrong;
        }
        return (right * 1.0) / (right + wrong);
    }
    
    double
    sq_mean_err(std::vector<std::vector<double>> &data)
    {
        std::vector<double> input_vals(num_inputs), target_vals(num_output);
        double err = 0;
        
        for (int i = 0; i < data.size(); ++i)
        {
            for (int j = 0; j < num_inputs; ++j)
                input_vals[j] = data[i][j];
            
            for (int j = num_inputs; j-num_inputs < num_output; ++j)
                target_vals[j-num_inputs] = data[i][j];
            
            std::vector<double> new_values = compute_values(input_vals);
            
            for (int q = 0; q < new_values.size(); ++q)
                err += (new_values[q] - target_vals[q]) * (new_values[q] - target_vals[q]);
        }
        
        return err / data.size();
    }
    
    std::vector<double>
    make_a_man_out_of_you(std::vector<std::vector<double>> &t_data, int max_epochs)
    {
        std::vector<double> hGradTerms(num_hidden);
        std::vector<double> oGradTerms(num_output);
        
        std::vector<std::vector<double>> hoWeightGradsAcc(num_hidden, std::vector<double>(num_output, 0));
        std::vector<std::vector<double>> ihWeightGradsAcc(num_inputs, std::vector<double>(num_hidden, 0));
        std::vector<double> oBiasGradsAcc(num_output);
        std::vector<double> hBiasGradsAcc(num_hidden);
        
        std::vector<std::vector<double>> hoPrevWeightGradsAcc(num_hidden, std::vector<double>(num_output, 0.01));
        std::vector<std::vector<double>> ihPrevWeightGradsAcc(num_inputs, std::vector<double>(num_hidden, 0.01));
        std::vector<double> oPrevBiasGradsAcc(num_output);
        std::vector<double> hPrevBiasGradsAcc(num_hidden);
        
        std::vector<std::vector<double>> hoPrevWeightDeltas(num_hidden, std::vector<double>(num_output, 0.01));
        std::vector<std::vector<double>> ihPrevWeightDeltas(num_inputs, std::vector<double>(num_hidden, 0.01));
        std::vector<double> oPrevBiasDeltas(num_output, 0.01);
        std::vector<double> hPrevBiasDeltas(num_hidden, 0.01);
        
        double etaPlus = 1.2; // values are from the paper
        double etaMinus = 0.5;
        double deltaMax = 50.0;
        double deltaMin = 0.000001;
        
        int epoch = 0;
        while (epoch < max_epochs)
        {
            ++epoch;
            
            if (epoch % 100 == 0)
            {
                double err = sq_mean_err(t_data);
                std::cout << "epoch = " << epoch << " err = " << err << std::endl;
            }
            
            // 1. compute and accumulate all gradients
            hoWeightGradsAcc = std::vector<std::vector<double>>(num_hidden, std::vector<double>(num_output, 0)); // zero-out values from prev iteration
            ihWeightGradsAcc = std::vector<std::vector<double>>(num_inputs, std::vector<double>(num_hidden, 0));
            oBiasGradsAcc = std::vector<double>(num_output, 0);
            hBiasGradsAcc = std::vector<double>(num_hidden, 0);
            
            auto xValues = std::vector<double>(num_inputs); // inputs
            auto tValues = std::vector<double>(num_output); // target values
            
            for (int row = 0; row < t_data.size(); ++row)  // walk thru all training data
            {
                for (int i = 0; i < num_inputs; ++i)
                    xValues[i] = t_data[row][i];
                
                for (int i = num_inputs; (i - num_inputs) < num_output; ++i)
                    tValues[i-num_inputs] = t_data[row][i];
                compute_values(xValues); // copy xValues in, compute outputs using curr weights (and store outputs internally)
                
                // compute the h-o gradient term/component as in regular back-prop
                // this term usually is lower case Greek delta but there are too many other deltas below
                
                for (int i = 0; i < num_output; ++i)
                {
                    double derivative = (1 - outputs[i]) * outputs[i]; // derivative of softmax = (1 - y) * y (same as log-sigmoid)
                    oGradTerms[i] = derivative * (outputs[i] - tValues[i]); // careful with O-T vs. T-O, O-T is the most usual
                }
                
                // compute the i-h gradient term/component as in regular back-prop
                
                for (int i = 0; i < num_hidden; ++i)
                {
                    double derivative = (1 - hidden_outputs[i]) * (1 + hidden_outputs[i]); // derivative of tanh = (1 - y) * (1 + y)
                    double sum = 0.0;
                    for (int j = 0; j < num_output; ++j) // each hidden delta is the sum of num_output terms
                    {
                        double x = oGradTerms[j] * hidden_output_weights[i][j];
                        sum += x;
                    }
                    hGradTerms[i] = derivative * sum;
                }
                
                // add input to h-o component to make h-o weight gradients, and accumulate
                
                for (int i = 0; i < num_hidden; ++i)
                {
                    for (int j = 0; j < num_output; ++j)
                    {
                        double grad = oGradTerms[j] * hidden_outputs[i];
                        hoWeightGradsAcc[i][j] += grad;
                    }
                }
                
                // the (hidden-to-) output bias gradients
                
                for (int i = 0; i < num_output; ++i)
                {
                    oBiasGradsAcc[i] += oGradTerms[i];
                }
                
                // add input term to i-h component to make i-h weight gradients and accumulate
                
                for (int i = 0; i < num_inputs; ++i)
                {
                    for (int j = 0; j < num_hidden; ++j)
                    {
                        double grad = hGradTerms[j] * inputs[i];
                        ihWeightGradsAcc[i][j] += grad;
                    }
                }
                
                // the (input-to-) hidden bias gradient
                
                for (int i = 0; i < num_hidden; ++i)
                {
                    hBiasGradsAcc[i] += hGradTerms[i];
                }
            } // each row
            // end compute all gradients
            
            // update all weights and biases (in any order)
            
            // update input-hidden weights
            double delta = 0.0;
            
            for (int i = 0; i < num_inputs; ++i)
            {
                for (int j = 0; j < num_hidden; ++j)
                {
                    if (ihPrevWeightGradsAcc[i][j] * ihWeightGradsAcc[i][j] > 0) // no sign change, increase delta
                    {
                        delta = ihPrevWeightDeltas[i][j] * etaPlus; // compute delta
                        if (delta > deltaMax) delta = deltaMax; // keep it in range
                        double tmp = -sign(ihWeightGradsAcc[i][j]) * delta; // determine direction and magnitude
                        input_hidden_weights[i][j] += tmp; // update weights
                        std::cout << "did it " << epoch << std::endl;
                    }
                    else if (ihPrevWeightGradsAcc[i][j] * ihWeightGradsAcc[i][j] < 0) // grad changed sign, decrease delta
                    {
                        delta = ihPrevWeightDeltas[i][j] * etaMinus; // the delta (not used, but saved for later)
                        if (delta < deltaMin) delta = deltaMin; // keep it in range
                        input_hidden_weights[i][j] -= ihPrevWeightDeltas[i][j]; // revert to previous weight
                        ihWeightGradsAcc[i][j] = 0; // forces next if-then branch, next iteration
                    }
                    else // this happens next iteration after 2nd branch above (just had a change in gradient)
                    {
                        delta = ihPrevWeightDeltas[i][j]; // no change to delta
                        // no way should delta be 0 . . .
                        double tmp = -sign(ihWeightGradsAcc[i][j]) * delta; // determine direction
                        input_hidden_weights[i][j] += tmp; // update
                    }
                    //Console.WriteLine(ihPrevWeightGradsAcc[i][j] + " " + ihWeightGradsAcc[i][j]); Console.ReadLine();
                    
                    ihPrevWeightDeltas[i][j] = delta; // save delta
                    ihPrevWeightGradsAcc[i][j] = ihWeightGradsAcc[i][j]; // save the (accumulated) gradient
                } // j
            } // i
            
            // update (input-to-) hidden biases
            for (int i = 0; i < num_hidden; ++i)
            {
                if (hPrevBiasGradsAcc[i] * hBiasGradsAcc[i] > 0) // no sign change, increase delta
                {
                    delta = hPrevBiasDeltas[i] * etaPlus; // compute delta
                    if (delta > deltaMax) delta = deltaMax;
                    double tmp = -sign(hBiasGradsAcc[i]) * delta; // determine direction
                    hidden_biases[i] += tmp; // update
                }
                else if (hPrevBiasGradsAcc[i] * hBiasGradsAcc[i] < 0) // grad changed sign, decrease delta
                {
                    delta = hPrevBiasDeltas[i] * etaMinus; // the delta (not used, but saved later)
                    if (delta < deltaMin) delta = deltaMin;
                    hidden_biases[i] -= hPrevBiasDeltas[i]; // revert to previous weight
                    hBiasGradsAcc[i] = 0; // forces next branch, next iteration
                }
                else // this happens next iteration after 2nd branch above (just had a change in gradient)
                {
                    delta = hPrevBiasDeltas[i]; // no change to delta
                    if (delta > deltaMax) delta = deltaMax;
                    else if (delta < deltaMin) delta = deltaMin;
                    // no way should delta be 0 . . .
                    double tmp = -sign(hBiasGradsAcc[i]) * delta; // determine direction
                    hidden_biases[i] += tmp; // update
                }
                hPrevBiasDeltas[i] = delta;
                hPrevBiasGradsAcc[i] = hBiasGradsAcc[i];
            }
            
            // update hidden-to-output weights
            for (int i = 0; i < num_hidden; ++i)
            {
                for (int j = 0; j < num_output; ++j)
                {
                    if (hoPrevWeightGradsAcc[i][j] * hoWeightGradsAcc[i][j] > 0) // no sign change, increase delta
                    {
                        delta = hoPrevWeightDeltas[i][j] * etaPlus; // compute delta
                        if (delta > deltaMax) delta = deltaMax;
                        double tmp = -sign(hoWeightGradsAcc[i][j]) * delta; // determine direction
                        hidden_output_weights[i][j] += tmp; // update
                    }
                    else if (hoPrevWeightGradsAcc[i][j] * hoWeightGradsAcc[i][j] < 0) // grad changed sign, decrease delta
                    {
                        delta = hoPrevWeightDeltas[i][j] * etaMinus; // the delta (not used, but saved later)
                        if (delta < deltaMin) delta = deltaMin;
                        hidden_output_weights[i][j] -= hoPrevWeightDeltas[i][j]; // revert to previous weight
                        hoWeightGradsAcc[i][j] = 0; // forces next branch, next iteration
                    }
                    else // this happens next iteration after 2nd branch above (just had a change in gradient)
                    {
                        delta = hoPrevWeightDeltas[i][j]; // no change to delta
                        // no way should delta be 0 . . .
                        double tmp = -sign(hoWeightGradsAcc[i][j]) * delta; // determine direction
                        hidden_output_weights[i][j] += tmp; // update
                    }
                    hoPrevWeightDeltas[i][j] = delta; // save delta
                    hoPrevWeightGradsAcc[i][j] = hoWeightGradsAcc[i][j]; // save the (accumulated) gradients
                } // j
            } // i
            
            // update (hidden-to-) output biases
            for (int i = 0; i < num_output; ++i)
            {
                if (oPrevBiasGradsAcc[i] * oBiasGradsAcc[i] > 0) // no sign change, increase delta
                {
                    delta = oPrevBiasDeltas[i] * etaPlus; // compute delta
                    if (delta > deltaMax) delta = deltaMax;
                    double tmp = -sign(oBiasGradsAcc[i]) * delta; // determine direction
                    output_biases[i] += tmp; // update
                }
                else if (oPrevBiasGradsAcc[i] * oBiasGradsAcc[i] < 0) // grad changed sign, decrease delta
                {
                    delta = oPrevBiasDeltas[i] * etaMinus; // the delta (not used, but saved later)
                    if (delta < deltaMin) delta = deltaMin;
                    output_biases[i] -= oPrevBiasDeltas[i]; // revert to previous weight
                    oBiasGradsAcc[i] = 0; // forces next branch, next iteration
                }
                else // this happens next iteration after 2nd branch above (just had a change in gradient)
                {
                    delta = oPrevBiasDeltas[i]; // no change to delta
                    // no way should delta be 0 . . .
                    double tmp = -sign(hBiasGradsAcc[i]) * delta; // determine direction
                    output_biases[i] += tmp; // update
                }
                oPrevBiasDeltas[i] = delta;
                oPrevBiasGradsAcc[i] = oBiasGradsAcc[i];
            }
        } // while
        
        auto wts = all_weights();
        return wts;
        
    }
    
};

#endif

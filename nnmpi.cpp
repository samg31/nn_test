#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <mpi.h>

using matrix = std::vector<std::vector<double>>;
using vec = std::vector<double>;

constexpr int num_inputs = 7;
constexpr int num_hidden = 9;
constexpr int num_output = 3;

constexpr double lower_bound = 0.0001;
constexpr double upper_bound = 0.01;

std::uniform_real_distribution<double> unif( lower_bound, upper_bound );
std::default_random_engine re;

vec inputs( num_inputs, 0.0 );
vec outputs( num_output );
vec hidden_biases( num_hidden );
vec hidden_outputs( num_hidden );
vec output_biases( num_output );

matrix ihWeights( num_inputs, vec( num_hidden, 0.0 ) );
matrix ohWeights( num_hidden, vec( num_output, 0.0 ) );

// fill vectors/matrices with a uniform random distribution
void initialize();

// determine if a number is positive. negative, or zero
int sign( double x );

// approximate tanh to 30 decimals
double tanh_approx( double x );
double sq_mean_error( matrix& data  );
vec train( matrix& t_data, int max_epochs );
vec get_weight();
vec soft_max( vec out_sums );
vec compute_values( vec in_vals );
int max_value_idx( vec v );

int main( int argc, char** argv )
{
    MPI_Init( &argc, &argv );
    int num_threads, rank;
    MPI_Comm_size( MPI_COMM_WORLD, &num_threads );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    if( rank == 0 )
    {
		bool converge = false;

		std::cout << "Initializing values... ";
		initialize();
		std::cout << "Completed." << std::endl;
		matrix input( 7018, vec( 10, 0.0 ) );

		std::ifstream reader("training.dat");
		std::string current;

		std::cout << "Reading data... ";
		for(int j = 0; j < input.size(); ++j)
		{
			for(int i = 0; i < 9; ++i)
			{
				getline(reader, current, ',');
				input[j][i] = std::stod(current);
			}
			getline(reader, current);
			input[j][9] = std::stod(current);
		}

		std::cout << "Completed." << std::endl;

		int max_epochs = 1000;


		std::cout << "Partitioning data table... ";
		std::size_t const partition = input.size() / (num_threads - 1);
		std::vector<matrix> partitioned_input;
		partitioned_input.reserve( num_threads - 1 );
		int j = 0;
		for( int i = 0; i < num_threads - 1; ++i )
		{
			matrix temp;
			for( ; j < partition*(i+1); ++j )
			{
				temp.push_back( input[j] );
			}
			partitioned_input.push_back( temp );
		}
		std::cout << "Completed." << std::endl;
		std::cout << "Sending partitions to slaves... ";
		for( int i = 0; i < partitioned_input.size(); ++i )
		{
			int size = partitioned_input[i].size();
			MPI_Send( &size, 1, MPI_INT, (i+1), 1, MPI_COMM_WORLD );
			for( auto& vector  : partitioned_input[i] )
				MPI_Send( &vector.front(), vector.size(), MPI_DOUBLE, (i+1), 0, MPI_COMM_WORLD );
		}
		std::cout << "Completed." << std::endl;
	
		// auto weights = train(input, max_epochs);

		// for(auto v : weights)
		//     std::cout << v << std::endl;
    }

    else
    {
		int size;
		MPI_Recv( &size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
		matrix table;
		vec row( 10 );
		table.reserve( size );
		for( int i = 0; i < size; ++i )
		{
			MPI_Recv( &row.front(), row.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
			table.push_back( row );
		}

    }
    MPI_Finalize();
    return 0;
}

void initialize()
{
    for( int i = 0; i < num_output; ++i )
    	output_biases[i] = unif(re);

    for( int i = 0; i < num_hidden; ++i )
		hidden_biases[i] = unif(re);

    for( int i = 0; i < num_hidden; ++i )
		for( int j = 0; j < num_output; ++j )
			ohWeights[i][j] = unif(re);

    for( int i = 0; i < num_inputs; ++i )
		for( int j = 0; j < num_hidden; ++j )
			ihWeights[i][j] = unif(re);
}

int sign( double x )
{
    if(x > 0)
		return 1;
    else if (x < 0)
		return -1;
    else
		return 0;    
}

double tanh_approx( double x )
{
    if (x < -20.0) return -1.0;
    else if (x > 20.0) return 1.0;
    else return std::tanh(x);
}

double sq_mean_error( matrix& data  )
{
    vec input_vals( num_inputs ), target_vals( num_output );
    double err = 0.0;

    for (int i = 0; i < data.size(); ++i)
    {
		for (int j = 0; j < num_inputs; ++j)
			input_vals[j] = data[i][j];
            
		for (int j = num_inputs; j-num_inputs < num_output; ++j)
			target_vals[j-num_inputs] = data[i][j];
            
		vec new_values = compute_values(input_vals);
            
		for (int q = 0; q < new_values.size(); ++q)
			err += (new_values[q] - target_vals[q]) * (new_values[q] - target_vals[q]);
    }
        
    return err / data.size();
}
vec compute_values( vec in_vals )
{
    vec hidden_sums( num_hidden, 0 ), out_sums( num_output, 0 );
        
    for (int i = 0; i < num_hidden; ++i)
		for (int j = 0; j < num_inputs; ++j)
			hidden_sums[i] += in_vals[j] * ihWeights[j][i];
        
    for (int i = 0; i < num_hidden; ++i)
		hidden_sums[i] += hidden_biases[i];
        
    for (int i = 0; i < num_hidden; ++i)
		hidden_outputs[i] = tanh_approx(hidden_sums[i]);
        
    for (int i = 0; i < num_output; ++i)
		for (int j = 0; j < num_hidden; ++j)
			out_sums[i] += hidden_outputs[j] * ohWeights[j][i];
        
    for (int i = 0; i < num_output; ++i)
		out_sums[i] += output_biases[i];
        
    outputs = soft_max(out_sums);
        
    return outputs;

}

vec soft_max( vec out_sums )
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

vec get_weight()
{
    int num_weights = (num_inputs * num_hidden)
		+ (num_hidden * num_output) + num_hidden + num_output;
    vec result( num_weights );
    int offset = ( num_inputs * num_hidden );

    for (int i = 0; i < ihWeights.size(); ++i)
		for (int j = 0; j < ihWeights[i].size(); ++j)
			result[(i * ihWeights.size()) + j] = ihWeights[i][j];
    for (int i = 0; i < hidden_biases.size(); ++i)
		result[(i * hidden_biases.size())+offset] = hidden_biases[i];
            
    offset += num_hidden;
            
    for (int i = 0; i < ohWeights.size(); ++i)
		for (int j = 0; j < ohWeights[0].size(); ++j)
			result[(i * ohWeights.size()) + j + offset] = ohWeights[i][j];
            
    offset += (num_hidden * num_output);
            
    for (int i = 0; i < output_biases.size(); ++i)
		result[(i * output_biases.size()) + offset] = output_biases[i];

    return result;

}

int max_value_idx( vec v )
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

vec train( matrix& t_data, int max_epochs )
{
    std::vector<double> hGradTerms(num_hidden);
    vec oGradTerms(num_output);
        
    matrix hoWeightGradsAcc(num_hidden, vec(num_output, 0));
    matrix ihWeightGradsAcc(num_inputs, vec(num_hidden, 0));
    vec oBiasGradsAcc(num_output);
    vec hBiasGradsAcc(num_hidden);
        
    matrix hoPrevWeightGradsAcc(num_hidden, vec(num_output, 0.01));
    matrix ihPrevWeightGradsAcc(num_inputs, vec(num_hidden, 0.01));
    vec oPrevBiasGradsAcc(num_output);
    vec hPrevBiasGradsAcc(num_hidden);
        
    matrix hoPrevWeightDeltas(num_hidden, vec(num_output, 0.01));
    matrix ihPrevWeightDeltas(num_inputs, vec(num_hidden, 0.01));
    vec oPrevBiasDeltas(num_output, 0.01);
    vec hPrevBiasDeltas(num_hidden, 0.01);
        
    double etaPlus = 1.2; // values are from the paper
    double etaMinus = 0.5;
    double deltaMax = 50.0;
    double deltaMin = 0.000001;
        
    int epoch = 0;
    while (epoch < max_epochs)
    {
		++epoch;
            
		if (epoch % 100 == 0 || epoch == 1 )
		{
			double err = sq_mean_error(t_data);
			std::cout << "epoch = " << epoch << " err = " << err << std::endl;
		}
            
		// 1. compute and accumulate all gradients
		hoWeightGradsAcc = matrix(num_hidden, vec(num_output, 0)); // zero-out values from prev iteration
		ihWeightGradsAcc = matrix(num_inputs, vec(num_hidden, 0));
		oBiasGradsAcc = vec(num_output, 0);
		hBiasGradsAcc = vec(num_hidden, 0);
            
		auto xValues = vec(num_inputs); // inputs
		auto tValues = vec(num_output); // target values
            
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
					double x = oGradTerms[j] * ohWeights[i][j];
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
					double grad = hGradTerms[j] * xValues[i];
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
				// std::cout << ihPrevWeightGradsAcc[i][j] << "*" << ihWeightGradsAcc[i][j] << std::endl;
				if (ihPrevWeightGradsAcc[i][j] * ihWeightGradsAcc[i][j] > 0) // no sign change, increase delta
				{
					delta = ihPrevWeightDeltas[i][j] * etaPlus; // compute delta
					if (delta > deltaMax) delta = deltaMax; // keep it in range
					double tmp = -sign(ihWeightGradsAcc[i][j]) * delta; // determine direction and magnitude
					ihWeights[i][j] += tmp; // update weights
					// std::cout << "did it " << epoch << std::endl;
				}
				else if (ihPrevWeightGradsAcc[i][j] * ihWeightGradsAcc[i][j] < 0) // grad changed sign, decrease delta
				{
					delta = ihPrevWeightDeltas[i][j] * etaMinus; // the delta (not used, but saved for later)
					if (delta < deltaMin) delta = deltaMin; // keep it in range
					ihWeights[i][j] -= ihPrevWeightDeltas[i][j]; // revert to previous weight
					ihWeightGradsAcc[i][j] = 0; // forces next if-then branch, next iteration
				}
				else // this happens next iteration after 2nd branch above (just had a change in gradient)
				{
					delta = ihPrevWeightDeltas[i][j]; // no change to delta
					// no way should delta be 0 . . .
					double tmp = -sign(ihWeightGradsAcc[i][j]) * delta; // determine direction
					ihWeights[i][j] += tmp; // update
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
					ohWeights[i][j] += tmp; // update
				}
				else if (hoPrevWeightGradsAcc[i][j] * hoWeightGradsAcc[i][j] < 0) // grad changed sign, decrease delta
				{
					delta = hoPrevWeightDeltas[i][j] * etaMinus; // the delta (not used, but saved later)
					if (delta < deltaMin) delta = deltaMin;
					ohWeights[i][j] -= hoPrevWeightDeltas[i][j]; // revert to previous weight
					hoWeightGradsAcc[i][j] = 0; // forces next branch, next iteration
				}
				else // this happens next iteration after 2nd branch above (just had a change in gradient)
				{
					delta = hoPrevWeightDeltas[i][j]; // no change to delta
					// no way should delta be 0 . . .
					double tmp = -sign(hoWeightGradsAcc[i][j]) * delta; // determine direction
					ohWeights[i][j] += tmp; // update
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
        
    auto wts = get_weight();
    return wts;
}

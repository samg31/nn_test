#include <iostream>
#include <fstream>
#include "neural_network.hpp"
#include <vector>

int main()
{

  std::vector<std::vector<double>> input(7018,std::vector<double>(10,0);
  std::ifstream reader("./training.dat");
  std::string current;
                                      
  for(int j = 0; j < input.size(); ++j)
  {
    for(int i = 0; i < 9; ++i)
    {
      getline(reader, current, ",");
      input[j][i] = std::stoi(current);
    }
    getline(reader, current);
    input[j][9] = std::stod(current);
  }

NeuralNetwork brain(10, 13, 3);
  int max_epochs = 1000;
  auto weights = brain.make_a_man_out_of_you(input, max_epochs);
  for(auto v : weights)
     std::cout << v << std::endl;

return 0;
}

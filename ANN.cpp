#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <chrono>

using namespace std;

///declare constants such as amount of hidden, input and output layers
const int hLayers = 20;
const int iLayers = 45;
const int oLayers = 5;
const float k = 1.0;
const float min_error = 0.004;
const float gain = 0.5;

///Declare prototypes for training and testing
void training(vector<int> &inputVec, vector<int> &targetVec, vector<float> &netVec, vector<float> &output, vector<float> &outputError, vector<float> &hOutput, vector<float> &errorVec, float(&weightMatrix)[hLayers][iLayers], float(&weightOutput)[hLayers][oLayers]);
void testing(vector<int> &inputVec, vector<int> &targetVec, vector<float> &netVec, vector<float> &output, vector<float> &hOutput, float(&weightMatrix)[hLayers][iLayers], float(&weightOutput)[hLayers][oLayers]);

int main()
{
	///declare vectors and arrays for input and calculation
	vector<int> inputVec;
	vector<int> targetVec;
	vector<float> netVec;
	vector<float> output;
	vector<float> outputError;
	vector<float> hOutput;
	vector<float> errorVec;
	float weightMatrix[hLayers][iLayers];
	float weightOutput[hLayers][oLayers];

	int menu = 0;

	do {
		cout << "Menu" << '\n';
		cout << "1. Training" << '\n';
		cout << "2. Testing" << '\n';
		cout << "3. End" << endl;
		cin >> menu;

		if (menu == 1) {
			training(inputVec, targetVec, netVec, output, outputError, hOutput, errorVec, weightMatrix, weightOutput);
		}
		else if (menu == 2) {
			testing(inputVec, targetVec, netVec, output, hOutput, weightMatrix, weightOutput);
		}
	} while (menu != 3);

	return 0;
}

void printInput(vector<int> &inputVec) {
	///Prints content of inputvector used for debugging and to for testing
	int gap = 0;
	for (int i = 1; i < inputVec.size(); i++) {
		cout << inputVec.at(i) << ' ';
		gap++;
		if (gap == 9) {
			cout << endl;
			gap = 0;
		}
	}
}

void fillWeights(float (&weightMatrix)[hLayers][iLayers], float (&weightOutput)[hLayers][oLayers]) {

	///Setup random number generator I found this to be the most reliable method
	std::default_random_engine generator;
	generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<double> distribution(-1, 1);

	///Randomly fill weights for input to hidden layer
	for (int x = 0; x < hLayers - 1; x++) {
		for (int y = 0; y < iLayers; y++) {
			weightMatrix[x][y] = distribution(generator);
		}
	}
	///Fill weights from hidden layer to output layer
	for (int x = 0; x < hLayers; x++) {
		for (int y = 0; y < oLayers; y++) {
			weightOutput[x][y] = distribution(generator);
		}
	}
}

void calcNet(vector<int> &inputVec, vector<float> &netVec, float (&weightMatrix)[hLayers][iLayers]) {
	float sum = 0;
	///Calculate Net for input into hidden layer ignoring the hidden bias
	for (int h = 0; h < hLayers - 1; h++) {
		sum = 0;
		for (int j = 0; j < iLayers; j++) {
			///This uses the input layer and the weights to each of the hidden layer to calculate the output
			sum += (inputVec.at(j)*weightMatrix[h][j]);
		}
		netVec.push_back(sum);
	}
}

void calcHiddenOutput(vector<float> &hOutput, vector<float> &netVec) {
	///Hidden node bias
	hOutput.push_back(1);
	///Calculate the actual output of the input to the hidden layer using the net from above
	for (int n = 0; n < netVec.size(); n++) {
		///Sigmoid function used as that is what was used in the lecture notes 5(a)
		hOutput.push_back(1.0 / (1.0 + (exp((-k)*netVec.at(n)))));
		///Fast sigmoid function
		//hOutput.push_back(netVec.at(n) / (1.0 + abs(netVec.at(n))));

	}
}

void calcOutput(vector<float> &hOutput, vector<float> &output, float (&weightOutput)[hLayers][oLayers]) {
	float sum = 0;
	///Calculate Net for input from hidden layer to output layer
	for (int o = 0; o < oLayers; o++) {
		sum = 0;
		for (int h = 0; h < hLayers; h++) {
			///Uses the hidden layer output and the weights to each output layer to calculate the actual output
			sum += (hOutput.at(h) * weightOutput[h][o]);
		}
		///Calculate actual output of input from hidden layers to output layer using the sigmoid function
		output.push_back(1.0 / (1.0 + (exp((-k)*sum))));
		///Fast sigmoid function
		//output = (sum / (1.0 + abs(sum)));
	}
}

void calcError(vector<float> &output, vector<float> &outputError, vector<int> &targetVec, vector<float> &errorVec, vector<float> &hOutput, float (&weightOutput)[hLayers][oLayers]) {
	float sum = 0;
	///Calculate Error value of input into output layer
	for (int o = 0; o < oLayers; o++) {
		///Uses actual output as well as the target output to calculate the error rate between them, this will be used to adjust the weights
		outputError.push_back((output.at(o) * (1 - output.at(o))) * (targetVec.at(o) - output.at(o)));
	}

	///Calulate error value of inputs into hidden layers
	for (int h = 1; h < hLayers; h++) {
		sum = 0;
		///Calculates the error of the hidden layer using the errorrate from the output layer as the hidden layer doesn't have an output
		for (int o = 0; o < oLayers; o++) {
			sum += (outputError.at(o)*weightOutput[h][o]);
		}
		errorVec.push_back((hOutput.at(h) * (1 - hOutput.at(h)) * sum));
	}
}

void updateWeights(float (&weightMatrix)[hLayers][iLayers], float (&weightOutput)[hLayers][oLayers], vector<float> &hOutput, vector<float> &outputError, vector<int> &inputVec, vector<float> &errorVec) {
	///Update weights from hidden layer to output layer
	for (int o = 0; o < oLayers; o++) {
		for (int h = 0; h < hLayers; h++) {
			///Adjusts the weights usingt the error from the output layer and the hidden layer
			weightOutput[h][o] = (weightOutput[h][o] + gain * hOutput.at(h) * outputError.at(o));
		}
	}

	///Update weights from input layers to hidden layers
	for (int x = 0; x < hLayers - 1; x++)
	{
		for (int j = 0; j < iLayers; j++)
		{
			///Adjusts the weights for the input layer using the input and error rate from the hidden layer
			weightMatrix[x][j] = (weightMatrix[x][j] + gain * inputVec.at(j) * errorVec.at(x));
		}
	}
}

void training(vector<int> &inputVec, vector<int> &targetVec, vector<float> &netVec, vector<float> &output, vector<float> &outputError, vector<float> &hOutput, vector<float> &errorVec, float(&weightMatrix)[hLayers][iLayers], float(&weightOutput)[hLayers][oLayers]) {
	bool flag = true;
	int epoch = 0;
	int patternCount = 0;
	float targetDif = 0;
	float avgError = 0;

	fillWeights(weightMatrix, weightOutput);

	while (flag) {
		ifstream infile;
		infile.open("training.txt");
		string line;

		int inputCount = 0;
		while (getline(infile, line)) {

			///Retrieve and store target and inputs
			stringstream stringS(line);
			while (stringS)
			{
				string inputS;
				getline(stringS, inputS, ' ');

				if (!inputS.empty()) {

					if (inputCount == 0) {
						inputVec.push_back(1);
					}

					int input = stof(inputS);
					///Input patterns
					if (inputCount < iLayers) {
						inputVec.push_back(input);
					}///Fill target vec using the targets inside the training file
					else if (inputCount >= iLayers) {
						targetVec.push_back(input);
					}
					inputCount++;
				}
			}

			if (inputCount == iLayers+5) {
				///Feedforward
				netVec.clear();
				hOutput.clear();
				errorVec.clear();

				calcNet(inputVec, netVec, weightMatrix);
				calcHiddenOutput(hOutput, netVec);
				calcOutput(hOutput, output, weightOutput);
				
				for (int o = 0; o < oLayers; o++) {
					targetDif += (targetVec.at(o) - output.at(o));
				}
				///Backprop
				calcError(output, outputError, targetVec, errorVec, hOutput, weightOutput);
				updateWeights(weightMatrix, weightOutput, hOutput, outputError, inputVec, errorVec);
				
				///Increase pattern count
				patternCount++;

				///Resets everything for next pattern
				inputCount = 0;
				inputVec.clear();
				targetVec.clear();
				output.clear();
				outputError.clear();
			}
		}

		///Calulcates the average error to determine when training is done
		avgError = targetDif / patternCount;
		if (avgError < 0) {
			avgError = avgError * -1;
		}
		cout << "Avg Error " << avgError << endl;
		patternCount = 0;
		targetDif = 0;
		epoch++;
		if (avgError < min_error || epoch >= 3000) {
			flag = false;
		}
		infile.close();
	}
	
	cout << "---EPOCH " << epoch << "---" << endl;

}

void testing(vector<int> &inputVec, vector<int> &targetVec, vector<float> &netVec, vector<float> &output, vector<float> &hOutput, float(&weightMatrix)[hLayers][iLayers], float(&weightOutput)[hLayers][oLayers]) {
	float sum = 0;

	ifstream infile;
	infile.open("testing.txt");
	string line;

	int inputCount = 0;
	while (getline(infile, line)) {
		///Get input and target from file
		stringstream stringS(line);
		while (stringS)
		{
			string inputS;
			getline(stringS, inputS, ' ');

			if (!inputS.empty()) {

				if (inputCount == 0) {
					inputVec.push_back(1);
				}

				int input = stof(inputS);
				///Input patterns
				inputVec.push_back(input);
				inputCount++;
			}
		}
		if (inputCount == iLayers) {
			///Feedforward
			netVec.clear();
			hOutput.clear();
			
			printInput(inputVec);
			cout << endl;

			calcNet(inputVec, netVec, weightMatrix);
			calcHiddenOutput(hOutput, netVec);
			calcOutput(hOutput, output, weightOutput);
			
			for (int o = 0; o < oLayers; o++) {
				///Step function for actual output
				cout << "Actual Output ";
				if (output.at(o) < 0.5) {
					cout << "0" << ' ';
				}
				else {
					///Makes the output easier to see
					cout << "1" << ' ';
					if (output.at(0) > 0.5) {
						cout << "Output is A " << ' ';
					}
					else if (output.at(1) > 0.5) {
						cout << "Output is E " << ' ';
					}
					else if (output.at(2) > 0.5) {
						cout << "Output is I " << ' ';
					}
					else if (output.at(3) > 0.5) {
						cout << "Output is O " << ' ';
					}
					else if (output.at(4) > 0.5) {
						cout << "Output is U " << ' ';
					}
				}
				cout << endl;

				
			}
		
			cout << endl;
			stringS.clear();
			inputCount = 0;
			inputVec.clear();
			targetVec.clear();
			output.clear();
		}

	}
	infile.close();
}
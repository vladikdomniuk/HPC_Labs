#include <iostream>
#include <cmath>
#include <mpi.h>

//Неперервна функція
double function(double x) {
    return x*x - cos(x) + exp(x);
}

// Функція обчислення визначеного інтеграла методом трапеції
double calculateIntegral(double start, double end, int stepsCount, int rank, int size) {
    double step = (end - start) / stepsCount; 
    double sum = 0.0;
    // Обчислення суми на кожному процесі
    if(stepsCount > 1)
    {
        for (int i = rank + 1; i <= stepsCount - 1; i += size) {
            sum += function(start + step * i);
        }
    }

    // Збір суми з усіх процесів
    double resultSum;
    MPI_Reduce(&sum, &resultSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Додавання функцій від меж інтеграла і визначення кінцевого результату на процесі 0
    if (rank == 0) {
        resultSum += (function(start) + function(end)) / 2.0;
        resultSum *= step;
    }

    return resultSum;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Перевірка наявності вказаних всіх параметрів
    if (argc != 4) {
        if (rank == 0)
            std::cout << "Invalid number of entered parameters.";
        MPI_Finalize();
        return 1;
    }

    double start = std::stod(argv[1]);  // Початок інтервалу
    double end = std::stod(argv[2]);  // Кінець інтервалу
    int stepsCount = std::stoi(argv[3]);     // Кількість кроків
    
    // Обчислення інтеграла на кожному процесі
    double integral = calculateIntegral(start, end, stepsCount, rank, size);

    //Вивід результату
    if (rank == 0)
        std::cout << "Integral result: " << integral << std::endl;

    MPI_Finalize();
    return 0;
}

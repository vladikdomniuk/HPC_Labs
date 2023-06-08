#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>

#define MATRIX_SIZE 3

//Друк матриці
void printMatrix(double** matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

//Ініціалізація матриці значеннями від 1 до 100
void initializeMatrix(double** matrix, int size) {
    //Для створення випадкових чисел при кожному виклику програми
    //srand(time(NULL)); 
    for (int i = 0; i < size; i++) {
        matrix[i] = new double[size];
        for (int j = 0; j < size; j++) {
            matrix[i][j] = rand() % 100 + 1;
        }
    }
}

void gramSchmidt(double** matrix, int size) {
    //#pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < size; j++) {
        //Ортогоналізація поточного стовпця відносно попередніх стовпців
        for (int k = 0; k < j; k++) {
            double dotMatrix = 0.0;
            // Обчислюємо скалярний добуток між j та k стовпцями
            //reduction(+:dotMatrix) вказує, що кожен потік буде обчислювати локально значення dotMatrix, а потім ці значення будуть просумовані
            #pragma omp parallel for reduction(+:dotMatrix)
            for (int i = 0; i < size; i++) {
                dotMatrix += matrix[i][k] * matrix[i][j];
            }
            
            //Віднімання проекції стовпця k від стовпця j
            #pragma omp parallel for
            for (int i = 0; i < size; i++) {
                matrix[i][j] -= dotMatrix * matrix[i][k];
            }
        }

        //Нормалізація отриманого ортогонального стовпця
        double norm = 0.0;
        #pragma omp parallel for reduction(+:norm)
        for (int i = 0; i < size; i++) {
            norm += matrix[i][j] * matrix[i][j];
        }

        norm = std::sqrt(norm);

        #pragma omp parallel for
        for (int i = 0; i < size; i++) {
            matrix[i][j] /= norm;
        }
    }
}

int main() {
    double** matrix = new double*[MATRIX_SIZE];
    initializeMatrix(matrix, MATRIX_SIZE);

    //Друк ініціалізованої матриці
    //std::cout << "Initialized matrix: " << std::endl;
    //printMatrix(matrix, MATRIX_SIZE);    

    std::cout << "Threads count:" << omp_get_max_threads() << "." << std::endl;

    //std::cout << "Start of algorithm execution." << std::endl;
    double startTime = omp_get_wtime();
    gramSchmidt(matrix, MATRIX_SIZE);
    double endTime = omp_get_wtime();
    //std::cout << "End of algorithm execution." << std::endl;

    std::cout << "Execution time: " << endTime - startTime << " seconds." << std::endl;

    //Друк результуючої матриці
    //std::cout << "Result matrix: " << std::endl;
    //printMatrix(matrix, MATRIX_SIZE);

    for (int i = 0; i < MATRIX_SIZE; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;

    return 0;
}
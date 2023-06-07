#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <omp.h>

#define MATRIX_SIZE 100
#define NUM_THREADS 8


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
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < size; i++) {
        matrix[i] = new double[size];
        for (int j = 0; j < size; j++) {
            matrix[i][j] = rand() % 100 + 1;
        }
    }
}

void gramSchmidt(double** matrix, int size) {
    for (int j = 0; j < size; j++) {
        double norm = 0.0;

        //Обчислення норми стовпця j
        #pragma omp parallel for reduction(+:norm) num_threads(NUM_THREADS)
        for (int i = 0; i < size; i++) {
            norm += matrix[i][j] * matrix[i][j];
        }
        
        norm = sqrt(norm);

        //Нормалізація стовпця j
        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < size; i++) {
            matrix[i][j] /= norm;
        }

        //Ортогоналізація наступних стовпців відносно стовпця j
        //shared(matrix, j) вказує, що змінні matrix і j є спільними для всіх потоків
        #pragma omp parallel for shared(matrix, j) num_threads(NUM_THREADS)
        for (int k = j + 1; k < size; k++) {
            double dotMatrix = 0.0;

            // Обчислюємо скалярний добуток між j-м та k-м стовпцями
            //reduction(+:dotMatrix) вказує, що кожен потік буде обчислювати локально значення dotMatrix, а потім ці значення будуть просумовані
            #pragma omp parallel for reduction(+:dotMatrix) num_threads(NUM_THREADS)
            for (int i = 0; i < size; i++) {
                dotMatrix += matrix[i][j] * matrix[i][k];
            }

            //Віднімання проекції стовпця j від стовпця k
            #pragma omp parallel for num_threads(NUM_THREADS)
            for (int i = 0; i < size; i++) {
                matrix[i][k] -= dotMatrix * matrix[i][j];
            }
        }
    }
}

int main() {
    double** matrix = new double*[MATRIX_SIZE];
    initializeMatrix(matrix, MATRIX_SIZE);

    //Друк ініціалізованої матриці
    //printMatrix(matrix, MATRIX_SIZE);
    //omp_get_max_threads()
    //std::cout << "Threads." << omp_get_max_threads() << std::endl;
    

    std::cout << "Start of algorithm execution." << std::endl;
    double startTime = omp_get_wtime();
    gramSchmidt(matrix, MATRIX_SIZE);
    double endTime = omp_get_wtime();
    std::cout << "End of algorithm execution." << std::endl;

    std::cout << "Execution time: " << endTime - startTime << " seconds." << std::endl;

    //Друк результуючої матриці
    //printMatrix(matrix, MATRIX_SIZE);

    for (int i = 0; i < MATRIX_SIZE; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;

    return 0;
}
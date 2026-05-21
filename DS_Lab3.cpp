#include <iostream>
#include <queue>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <windows.h>
#include <chrono>


using namespace std;

struct Point {
    int x, y;
};

class ArrayQueue {
private:
    Point* data;
    int head, tail, capacity, count;
public:
    ArrayQueue(int size) : capacity(size), head(0), tail(0), count(0) {
        data = new Point[capacity];
    }
    ~ArrayQueue() { delete[] data; }

    void push(Point p) {
        if (count < capacity) {
            data[tail] = p;
            tail = (tail + 1) % capacity;
            count++;
        }
    }

    void pop() {
        if (count > 0) {
            head = (head + 1) % capacity;
            count--;
        }
    }

    Point front() { return data[head]; }
    bool empty() { return count == 0; }
};

struct Node {
    Point p;
    Node* next;
};

class ListQueue {
private:
    Node* head = nullptr, * tail = nullptr;
public:
    void push(Point val) {
        Node* temp = new Node{ val, nullptr };
        if (!tail) {
            head = tail = temp;
        }
        else {
            tail->next = temp;
            tail = temp;
        }
    }

    void pop() {
        if (head) {
            Node* temp = head;
            head = head->next;
            if (!head) tail = nullptr;
            delete temp;
        }
    }

    Point front() { return head->p; }
    bool empty() { return head == nullptr; }
};

template <typename T>
int countComponents(int M, int N, vector<vector<bool>>& grid, bool isCylinder, T& q) {
    int components = 0;
    vector<vector<bool>> visited(M, vector<bool>(N, false));

    int dx[] = { 0, 0, 1, -1 };
    int dy[] = { 1, -1, 0, 0 };

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (grid[i][j] && !visited[i][j]) {
                components++;

                q.push({ i, j });
                visited[i][j] = true;

                while (!q.empty()) {
                    Point curr = q.front();
                    q.pop();

                    for (int k = 0; k < 4; ++k) {
                        int nx = curr.x + dx[k];
                        int ny = curr.y + dy[k];

                        if (isCylinder) {
                            nx = (nx + M) % M;
                        }

                        if (nx >= 0 && nx < M && ny >= 0 && ny < N) {
                            if (grid[nx][ny] && !visited[nx][ny]) {
                                visited[nx][ny] = true;
                                q.push({ nx, ny });
                            }
                        }
                    }
                }
            }
        }
    }
    return components;
}

void printGrid(const vector<vector<bool>>& grid) {
    for (const auto& row : grid) {
        for (bool cell : row) {
            std::cout << (cell ? "1 " : "0 ");
        }
        std::cout << "\n";
    }
}

int main(){
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
    srand(time(NULL));

    int M, N;
    bool isCylinder;

    cout << "Введите M и N: "; cin >> M >> N;

    vector<vector<bool>> grid(M, vector<bool> (N));

    for (int i = 0; i < M; ++i) {
        generate(grid[i].begin(), grid[i].end(), []() {return rand() % 2;});
    }

    cout << "Лист:" << endl;
    printGrid(grid);

    cout << "Склеили ли лист в цилиндр? 0 -> Нет; 1 -> Да" << endl; cin >> isCylinder;
    
    int k1, k2, k3;
        
        ArrayQueue q1(M*N);
        ListQueue q2;
        queue<Point> q3;

        cout << "Очередь на массиве. ";
        auto start = chrono::high_resolution_clock::now();
        k1 = countComponents(M, N, grid, isCylinder, q1);
        auto end = chrono::high_resolution_clock::now();
        cout << "Время выполнения: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " mсs" << endl;

        cout << "Очередь на связанном списке. ";
        start = chrono::high_resolution_clock::now();
        k2 = countComponents(M, N, grid, isCylinder, q2);
        end = chrono::high_resolution_clock::now();
        cout << "Время выполнения: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " mсs" << endl;

        cout << "STL-очередь. ";
        start = chrono::high_resolution_clock::now();
        k3 = countComponents(M, N, grid, isCylinder, q3);
        end = chrono::high_resolution_clock::now();
        cout << "Время выполнения: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " mсs" << endl;

        if (k1 == k2 && k2 == k3 && k3 == k1) cout << "Количество кусков: " << k1 << endl;
        else cout << "Количество кусков не совпадает!" << endl;

    return 0;
}
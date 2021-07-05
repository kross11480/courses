#include <iostream>

class Array {
  // Array contains integers and is given a size by user
  private:
    int m_size;
    int* m_data;
    void copy_into(const Array& obj) {
      // Creating a copy
      for (int i = 0; i < m_size; i++) {
        m_data[i] = obj.m_data[i];
      }
    }
  public:
    // Constructor
    Array(int len) : m_size(len), m_data(new int[len]) {}
    // Destructor
    ~Array() { delete[] m_data; }
    // Copy Constructor
    Array(const Array& obj) : m_size(obj.m_size), m_data(new int[obj.m_size]) {
      copy_into(obj);
    }
    // Assignment Operator
    Array& operator= (Array const& obj) {
      m_size = obj.m_size;
      m_data = obj.m_data;
      // copy_into(obj);
      return *this;
    }
    // Subscript operator
    int& operator[] (int index) {
      return m_data[index];
    }
};

int main(int argc, char *argv[]) {
  // Test constructor
  Array* arr_ptr = new Array(3);
  Array& arr = *arr_ptr;
  // Test subscript operator
  std::cout << arr[0] << std::endl;
  std::cout << arr[1] << std::endl;
  std::cout << arr[2] << std::endl;
  arr[0] = 1;
  arr[1] = 2;
  arr[2] = 3;
  std::cout << arr[0] << std::endl;
  std::cout << arr[1] << std::endl;
  std::cout << arr[2] << std::endl;
  // Test copy constructor
  Array arr1(arr);
  std::cout << arr1[2] << std::endl;
  // Test assignment operator
  Array arr2(1);
  arr2 = arr;
  delete arr_ptr;
  // std::cout << arr2[2] << std::endl;
  //std::cout << "Hello World!" << std::endl;
  return 0;
}



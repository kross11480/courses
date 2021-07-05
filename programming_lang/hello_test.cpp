#include <iostream>
#include "Array.h"

template<typename T>
void swap(T& i, T& j){
  T tmp = j;
  j = i;
  i = tmp;
}

int main(int argc, char *argv[]) {
  // Test constructor
  Array<float> arr(3);
  // Test subscript operator
  arr[0] = 1.0;
  arr[1] = 2.0;
  arr[2] = 3.5;
  std::cout << arr[0] << std::endl;
  std::cout << arr[1] << std::endl;
  std::cout << arr[2] << std::endl;
  // Test copy constructor
  Array<float> arr1(arr);
  std::cout << arr1[2] << std::endl;
  // Test assignment operator
  Array<float> arr2(1);
  arr2 = arr;
  std::cout << arr2[2] << std::endl;
  //std::cout << "Hello World!" << std::endl;
  int i,j = 5;
  long a,b = 4;
  swap<Array<float>>(arr1,arr2);
  swap<long>(a,b);
  return 0;
}

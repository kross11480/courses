template<typename T>
class Array {
  // Array contains integers and is given a size by user
  private:
    int m_size;
    T* m_data;
    void copy_into(const Array& obj) {
      // Creating a copy
      for (int i = 0; i < m_size; i++) {
        m_data[i] = obj.m_data[i];
      }
    }
  public:
    // Constructor
    Array(int len) : m_size(len), m_data(new T[len]) {}
    // Destructor
    ~Array() { delete[] m_data; }
    // Copy constructor
    Array(const Array<T>& obj) : m_size(obj.m_size), m_data(new T[obj.m_size]) {
      copy_into(obj);
    }
    // Copy assignment operator
    Array<T>& operator= (Array<T> const& obj) {
      m_size = obj.m_size;
      copy_into(obj);
      return *this;
    }

    const T& operator[] (int) const;
    T& operator[] (int);

    int len() const {
      //m_size = 10;
      return m_size;
    }
};

// Subscript operator
template<typename T>
const T& Array<T>::operator[] (int index) const {
  return m_data[index];
}

template<typename T>
T& Array<T>::operator[] (int index) {
  return m_data[index];
}

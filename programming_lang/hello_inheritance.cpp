#include <iostream>

// Base Class: Shape, destructor, draw
class Shape {
public:
    virtual ~Shape() {
        std::cout << ("Shape::destructor") << std::endl;
    }
    // virtual void draw() = 0;
    virtual void draw() {
        std::cout << ("Shape::draw") << std::endl;
    }
};

// Derived Class: Circle,
class Circle : public Shape {
public:
    ~Circle() {
        std::cout << ("Circle::destructor") << std::endl;
    }
    void draw() {
        std::cout << ("Circle::draw") << std::endl;
    }
};

int main() {
    Shape *shape = new Shape;
    Circle *circle = new Circle;
    Shape *sh = circle;
    shape->draw();
    circle->draw();
    sh->draw();
    delete sh;
    return 0;
}

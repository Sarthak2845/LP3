// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentData {

    struct Student {
        uint rollNo;
        string name;
        uint marks;
    }

    Student[] public students;

    // Add a student
    function addStudent(uint _rollNo, string memory _name, uint _marks) public {
        students.push(Student(_rollNo, _name, _marks));
    }

    // Get one student by index
    function getStudent(uint index) public view returns (uint, string memory, uint) {
        Student memory s = students[index];
        return (s.rollNo, s.name, s.marks);
    }

    // Total number of students
    function getCount() public view returns (uint) {
        return students.length;
    }

    // Runs when a function that doesn't exist is called
    fallback() external payable {}

    // Runs when plain Ether is sent
    receive() external payable {}
}
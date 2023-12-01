/**
Author: David GC
Date: 30 November 2023
Problem : Leetcode 
 */

var happyNumber = function (n) {
  // create a set to keep the track if a cycle is detected
  let visitedNumber = new Set();
  function recurse(num) {
    //base condition to exit the recursion or the number is a happy number
    if (num == 1) {
      return true;
    }

    if (visitedNumber.has(num)) {
      return false;
    }

    visitedNumber.add(num);

    let squaredSum = 0;
    let arrayNum = [...num.toString()].map(Number);

    for (let i = 0; i < arrayNum.length; i++) {
      squaredSum += arrayNum[i] * arrayNum[i];
    }

    return recurse(squaredSum);
  }

  return recurse(n);
};

// console.log(happyNumber(2));

// Leetcode problem 112: Path sum

var pathSum = function (root, targetSum) {
  //dfs recursion
  function recurse(root, currSum) {
    //base case
    if (root === null) {
      return false;
    }

    currSum += root.val;

    if (!root.left && !root.right) {
      //if there is no left and right leaf node then we compare the current sum to the target sum
      return currSum === targetSum;
    }

    return recurse(root.left, currSum) || recurse(root.right, currSum);
  }

  return recurse(root, 0);
};

// Leetcode problem 67: Add binary

var addBinary = function (a, b) {
  //create a result var as a string
  let result = "";

  let i = a.length - 1;
  let j = b.length - 1;
  let carry = 0;

  //loop through the binary digits from right to left
  while (i >= 0 || j >= 0) {
    let sum = carry;
    //sum of two bits
    if (i >= 0) {
      //If there are more bits in a, add the current bit to the sum
      sum += a[i--] - "0";
    }

    if (j >= 0) {
      //If there are more bits in a, add the current bit to the sum
      sum += b[j--] - "0";
    }

    //Add the bit to the result
    result = (sum % 2) + result;

    //modify carry
    carry = parseInt(sum / 2);
  }

  //check for the final time if there is any carry
  if (carry > 0) {
    result = 1 + result;
  }

  return result;
};

addBinary("11", "1");

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

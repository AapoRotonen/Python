def avg(numberArray):
    total = 0  # Initialize sum
    for numbers in numberArray:  # Iterate through the array
        total += numbers  # Add each number to the total
          
    return total / len(numberArray)  # Calculate and return the average

# Example usage
numbers = [10, 20, 30, 40, 50]
print(avg(numbers))  # Output: 30.0

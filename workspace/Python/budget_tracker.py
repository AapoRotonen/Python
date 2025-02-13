import os
import json
from tabulate import tabulate

# File to store data
DATA_FILE = "budget_data.json"

# Load data from file (if exists)
def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as file:
            return json.load(file)
    else:
        return {"income": 0, "expenses": [], "balance": 0}

# Save data to file
def save_data(data):
    with open(DATA_FILE, "w") as file:
        json.dump(data, file, indent=4)

# Add income to the budget
def add_income(data, amount):
    data["income"] += amount
    data["balance"] += amount
    save_data(data)

# Add expense to the budget
def add_expense(data, amount, category):
    data["expenses"].append({"amount": amount, "category": category})
    data["balance"] -= amount
    save_data(data)

# View current balance
def view_balance(data):
    print(f"Current Balance: ${data['balance']:.2f}")

# View expense history
def view_expenses(data):
    if data["expenses"]:
        print("\nExpense History:")
        # Dynamically get headers from the first expense dictionary
        headers = data["expenses"][0].keys()  # Get the keys (headers)
        # Display the expenses using tabulate
        print(tabulate(data["expenses"], headers=headers, tablefmt="grid"))
    else:
        print("No expenses recorded yet.")

# Main Menu
def menu():
    print("\n=== Budget Tracker ===")
    print("1. Add Income")
    print("2. Add Expense")
    print("3. View Balance")
    print("4. View Expenses")
    print("5. Exit")

def main():
    data = load_data()

    while True:
        menu()
        choice = input("Choose an option: ")

        if choice == "1":
            amount = float(input("Enter income amount: "))
            add_income(data, amount)
            print(f"Added ${amount:.2f} income.")
        elif choice == "2":
            amount = float(input("Enter expense amount: "))
            category = input("Enter expense category: ")
            add_expense(data, amount, category)
            print(f"Added ${amount:.2f} expense in category: {category}.")
        elif choice == "3":
            view_balance(data)
        elif choice == "4":
            view_expenses(data)
        elif choice == "5":
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()

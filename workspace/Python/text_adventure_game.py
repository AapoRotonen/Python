import random
import time

# Game classes

class Player:
    def __init__(self, name):
        self.name = name
        self.health = 100
        self.attack = 10
        self.inventory = []
        self.quest_log = []
        self.skills = {"Combat": 1, "Fishing": 1, "Mining": 1, "Woodcutting": 1, "Firemaking": 1}
        self.location = "Lumbridge"

    def take_damage(self, amount):
        self.health -= amount
        if self.health < 0:
            self.health = 0

    def heal(self, amount):
        self.health += amount
        if self.health > 100:
            self.health = 100

    def is_alive(self):
        return self.health > 0

    def attack_enemy(self, enemy):
        damage = random.randint(self.attack - 3, self.attack + 3)
        enemy.take_damage(damage)
        print(f"{self.name} attacks {enemy.name} for {damage} damage!")

    def __str__(self):
        return f"{self.name} - Health: {self.health} - Attack: {self.attack} - Location: {self.location}"


class Enemy:
    def __init__(self, name, health, attack):
        self.name = name
        self.health = health
        self.attack = attack

    def take_damage(self, amount):
        self.health -= amount
        if self.health < 0:
            self.health = 0

    def is_alive(self):
        return self.health > 0

    def attack_player(self, player):
        damage = random.randint(self.attack - 2, self.attack + 2)
        player.take_damage(damage)
        print(f"{self.name} attacks {player.name} for {damage} damage!")

    def __str__(self):
        return f"{self.name} - Health: {self.health} - Attack: {self.attack}"


# Game functions

def choose_name():
    name = input("What is your adventurer's name? ")
    return name


def intro_story():
    print("\nWelcome to the world of Gielinor!")
    print("Choose your adventure and prepare for a journey through lands inspired by Old School RuneScape!")
    print("You'll start in Lumbridge, but the choices are endless.")
    time.sleep(2)


def start_adventure():
    # Player setup
    name = choose_name()
    player = Player(name)

    # Game Intro
    intro_story()

    # Start Adventure
    lumbridge_adventure(player)


def lumbridge_adventure(player):
    print(f"\n{player.name}, your journey begins in Lumbridge!")
    time.sleep(1)

    while True:
        print(f"\nYou are in Lumbridge, your current health is {player.health} HP.")
        print("You can choose what to do:")
        print("1. Help a farmer with a goblin quest")
        print("2. Go fishing at Lumbridge Swamp")
        print("3. Try woodcutting in the nearby forest")
        print("4. Explore Varrock")
        print("5. Check your inventory and quest log")
        print("6. Exit the game (type 'exit')")

        choice = input("What would you like to do? ")

        if choice == "1":
            print("\nYou decide to help the farmer and venture into the goblin cave!")
            enemy = Enemy("Goblin", 30, 5)
            battle(player, enemy)
            if player.is_alive():
                print(f"\nYou have defeated the Goblin!")
                player.quest_log.append("Goblin Slayer")
                print(f"Quest completed: Goblin Slayer")
                time.sleep(1)
            else:
                print(f"\n{player.name} has been defeated! Game over.")
                break

        elif choice == "2":
            fishing(player)
        elif choice == "3":
            woodcutting(player)
        elif choice == "4":
            varrock_adventure(player)
        elif choice == "5":
            print(f"\nInventory: {player.inventory}")
            print(f"Quests completed: {player.quest_log}")
            time.sleep(1)
        elif choice.lower() == 'exit':
            print("\nThanks for playing!")
            break
        else:
            print("\nInvalid choice. Try again.")


def fishing(player):
    print("\nYou head to the Lumbridge Swamp and find a fishing spot.")
    while True:
        choice = input("Do you (1) try fishing, (2) leave the spot, or (3) return to Lumbridge? ")

        if choice == "1":
            print("\nYou cast your fishing rod into the water, waiting patiently.")
            fish = random.choice(["Raw Salmon", "Raw Trout", "Nothing"])
            if fish != "Nothing":
                player.inventory.append(fish)
                print(f"\nYou caught a {fish}!")
                print(f"Your inventory: {player.inventory}")
                player.skills["Fishing"] += 1
                print(f"Your Fishing skill is now level {player.skills['Fishing']}.")
                time.sleep(1)
            else:
                print("\nYou didn't catch anything.")
                time.sleep(1)

        elif choice == "2":
            print("\nYou leave the fishing spot.")
            break

        elif choice == "3":
            print("\nYou return to Lumbridge.")
            break

        else:
            print("Invalid choice. Try again.")


def woodcutting(player):
    print("\nYou find a forest full of trees and decide to try your hand at woodcutting.")
    while True:
        choice = input("Do you (1) chop a tree, (2) leave the forest, or (3) return to Lumbridge? ")

        if choice == "1":
            if player.skills["Woodcutting"] < 5:
                print("\nYou struggle to chop the tree, but you manage to collect some logs.")
                player.inventory.append("Logs")
                player.skills["Woodcutting"] += 1
                print(f"Your Woodcutting skill is now level {player.skills['Woodcutting']}.")
            else:
                print("\nYou chop down a large tree and gather several logs. Your skill has improved!")
                player.inventory.append("Logs")
                player.skills["Woodcutting"] += 2
                print(f"Your Woodcutting skill is now level {player.skills['Woodcutting']}.")
            time.sleep(1)

        elif choice == "2":
            print("\nYou leave the forest and continue your journey.")
            break

        elif choice == "3":
            print("\nYou return to Lumbridge.")
            break

        else:
            print("Invalid choice. Try again.")


def varrock_adventure(player):
    print(f"\n{player.name}, your journey takes you to the bustling city of Varrock!")
    print("You walk down the busy streets, seeing traders selling goods and citizens talking.")
    while True:
        print("\nWhat would you like to do in Varrock?")
        print("1. Visit the Grand Exchange")
        print("2. Explore the marketplace")
        print("3. Visit the castle")
        print("4. Return to Lumbridge")

        choice = input("What will you do? ")

        if choice == "1":
            print("\nYou head to the Grand Exchange to see the market prices.")
            event = random.choice(["Thief steals your coin pouch!", "Merchant gives you a discount on potions."])
            if event == "Thief steals your coin pouch!":
                print("\nA thief snatches your coin pouch! You lose some gold.")
                time.sleep(1)
            else:
                print("\nA merchant offers you a discount on potions. You gain some healing potions!")
                player.inventory.append("Healing Potion")
                print(f"Your inventory: {player.inventory}")
                time.sleep(1)

        elif choice == "2":
            print("\nYou explore the marketplace and find a mysterious old man selling a rare artifact.")
            choice = input("Do you (1) buy the artifact, or (2) walk away? ")
            if choice == "1":
                print("\nThe artifact is cursed! You are transported to a strange new world!")
                time.sleep(2)
            elif choice == "2":
                print("\nYou decide not to buy the artifact and leave the marketplace.")
                time.sleep(1)

        elif choice == "3":
            print("\nYou visit the castle and talk to the king. He offers you a quest to retrieve a stolen gem.")
            player.quest_log.append("Retrieve the Gem")
            print(f"Quest log updated: Retrieve the Gem")
            time.sleep(1)

        elif choice == "4":
            print("\nYou decide to return to Lumbridge.")
            break

        else:
            print("Invalid choice. Try again.")


def battle(player, enemy):
    while player.is_alive() and enemy.is_alive():
        print(f"\n{player.name}: {player.health} HP | {enemy.name}: {enemy.health} HP")
        action = input("Do you (1) attack, (2) heal? ")

        if action == "1":
            player.attack_enemy(enemy)
            if enemy.is_alive():
                enemy.attack_player(player)
            else:
                print(f"\n{enemy.name} is defeated!")
                break
        elif action == "2":
            player.heal(10)
            print(f"\nYou heal yourself! Your health is now {player.health}.")
            enemy.attack_player(player)
        else:
            print("Invalid choice. You missed your turn!")
            enemy.attack_player(player)
        
        time.sleep(1)


# Start the game
if __name__ == "__main__":
    start_adventure()

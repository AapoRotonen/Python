import requests
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Get the API_KEY from the environment variables
API_KEY = os.getenv('API_KEY')

# OpenWeatherMap API URL
BASE_URL = 'https://api.openweathermap.org/data/2.5/weather'

# Check if API_KEY exists
if not API_KEY:
    print("API key is missing! Please check the .env file.")
    exit(1)

# Function to get weather information for a city
def get_weather(city):
    url = f"{BASE_URL}?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)

    # Check if the API request was successful
    if response.status_code == 200:
        data = response.json()
        
        # Extract weather information from the response data
        city_name = data.get('name', 'Unknown city')
        temperature = data['main'].get('temp', 'N/A')
        humidity = data['main'].get('humidity', 'N/A')
        weather_description = data['weather'][0].get('description', 'No description available')
        
        print(f"\nWeather in {city_name}:")
        print(f"Temperature: {temperature}°C")
        print(f"Humidity: {humidity}%")
        print(f"Description: {weather_description}")
    else:
        # If the request failed, print an error message
        print(f"Error fetching weather data for {city}. Status Code: {response.status_code}")
        print(f"Error Message: {response.json().get('message', 'Unknown error')}")

# Main function to interact with the user
def main():
    print("Welcome to the Weather API Caller!")
    while True:
        city = input("\nEnter a city name (or type 'exit' to quit): ")
        if city.lower() == 'exit':
            print("Goodbye!")
            break
        get_weather(city)

if __name__ == "__main__":
    main()

# allows usage of random, sys, and time libraries
import random, sys, time

# creates the board
hitBoard = []
enemyShipBoard = []

# defines names of the ships
corvetteNames = ['Augusta', 'Canberra', 'Charleston', 'Cincinnati', 'Jackson', 'Kansas City', 'Manchester', 'Mobile', 'Montgomery', 'Oakland']
destroyerNames = ['Bainbridge', 'Chafee', 'Cole', 'Higgins', "Hopper"]
cruiserNames = ['Antietam', 'Chosin', 'Cowpens', 'Gettysburg', 'Normandy']
submarineNames = ['Annapolis', 'Alexandria']
frigateNames = ['Constellation', 'Congress']
battlecruiserNames = ['Alaska', 'Hawaii']
battleshipNames = ['Missouri']
aircraftCarrierNames = ['Gerald R Ford']

# creates the board with size 25
def createBoard(size):
    for i in range(size):
        hitBoard.append(['0'] * size)
        enemyShipBoard.append(['0'] * size)

createBoard(25)

# creates the screen
def newScreen():
    print('\nBattleship')
    print('----------')
    print('                       1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2')
    print('     1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5')
    print('   +--------------------------------------------------')
    printBoard(hitBoard)

# creates the board
def printBoard(board): 
    idx = 1
    for row in board:
        if idx >= 10:
            toPrint = str(idx)
        else:
            toPrint = ' ' + str(idx)
        toPrint += ' | '
        toPrint += ' '.join(row)
        print(toPrint)
        idx += 1
    print('\n')

# creates a tuple of 2 random positions on the board
def randomPosition(board, xBound, yBound):
    xCord = random.randint(0, len(board) - xBound)
    yCord = random.randint(0, len(board[0]) - yBound)
    return (xCord, yCord)

# creates the ship
class Ship:
    def __init__(self, type, size, name):
        self.size = size
        self.type = type
        self.name = name
        self.dir = random.randint(0, 1) * 90
        self.positions = []
        self.damage = 0

        emptySpace = False
        row = 0
        col = 0

        # places the ship
        while not emptySpace:
            emptySpace = True
            if self.dir == 0:
                (row, col) = randomPosition(hitBoard, self.size, 1)
                for i in range(self.size):
                    if enemyShipBoard[row + i][col] == '1':
                        emptySpace = False
                        break
            elif self.dir == 90:
                (row, col) = randomPosition(hitBoard, 1, self.size)
                for i in range(self.size):
                    if enemyShipBoard[row][col + i] == '1':
                        emptySpace = False
                        break
        # adds the ship to the list of ships
        if self.dir == 0:
            for i in range(size):
                self.positions.append([row + i, col])
                enemyShipBoard[row + i][col] = '1'
        else:
            for i in range(size):
                self.positions.append([row, col + i])
                enemyShipBoard[row][col + i] = '1'

# creates the ships
ships = []
for i in range(10):
    name = corvetteNames[random.randint(0, len(corvetteNames) - 1)]
    ships.append(Ship('Corvette', 2, name))
    corvetteNames.remove(name)
for i in range(5):
    name = destroyerNames[random.randint(0, len(destroyerNames) - 1)]
    ships.append(Ship('Destroyer', 3, name))
    destroyerNames.remove(name)
for i in range(5):
    name = cruiserNames[random.randint(0, len(cruiserNames) - 1)]
    ships.append(Ship('Cruiser', 5, name))
    cruiserNames.remove(name)
for i in range(2):
    name = submarineNames[random.randint(0, len(submarineNames) - 1)]
    ships.append(Ship('Submarine', 4, name))
    submarineNames.remove(name)
for i in range(2):
    name = frigateNames[random.randint(0, len(frigateNames) - 1)]
    ships.append(Ship('Frigate', 2, name))
    frigateNames.remove(name)
for i in range(2):
    name = battlecruiserNames[random.randint(0, len(battlecruiserNames) - 1)]
    ships.append(Ship('Battlecruiser', 6, name))
    battlecruiserNames.remove(name)
ships.append(Ship('Battleship', 7, battleshipNames[0]))
ships.append(Ship('Aircraft Carrier', 9, aircraftCarrierNames[0]))

# finds out if a string can be turned into an integer
def isInt(str):
    try:
        int(str)
    except:
        return False
    return True

# gathers the input and stops the script if you say exit, stop or kill
def inputInt(prompt: str):
    answer = input(prompt)
    answer = answer.lower()
    if answer == 'exit' or answer == 'stop' or answer == 'kill' or answer == 'quit' or answer == 'q':
        sys.exit()
    if not isInt(answer):
        return None
    if int(answer) < 1 or int(answer) > len(hitBoard):
        return None
    return int(answer) - 1

# main loop
def main():
    guesses = 0
    startTime = time.time()
    while len(ships) != 0:
        print(hitBoard)
        # handles losing
        if guesses > 500:
            print('You spent all 500 guesses and took {} seconds. You lose!'.format(time.time()-startTime))
            sys.exit()
        # handles player input
        guesses += 1
        newScreen()
        guessRow = inputInt('Enter a Row: ')
        guessCol = inputInt('Enter Column: ')
        if guessRow == None or guessCol == None:
            print('That\'s not even an option')
        elif guessRow < 0 or guessRow > len(
                hitBoard) - 1 or guessCol < 0 or guessCol > len(
                    hitBoard[0]) - 1:
            print('That\'s not even an option')
        elif hitBoard[guessRow][guessCol] != '0':
            print('You already shot there')
        else:
            # manages hitting ships
            hit = False
            for ship in ships:
                for pos in ship.positions:
                    if guessRow == pos[0] and guessCol == pos[1]:
                        hit = True
                        ship.damage += 1
                        if ship.damage == ship.size:
                            for section in ship.positions:
                                hitBoard[section[0]][section[1]] = 'S'
                            ships.remove(ship)
                            if len(ships) == 0:
                                break
                            print(
                                'You sank the {} {}. Don\'t worry, I still have {} ships left.'
                                .format(ship.type, ship.name, len(ships)))
                        else:
                            print('Well, I guess you\'ve hit the {} {}'.format(
                                ship.type, ship.name))
                            hitBoard[guessRow][guessCol] = 'H'
                        break
                if hit:
                    break
            if not hit:
                # handles misses
                print('You missed!')
                hitBoard[guessRow][guessCol] = 'X'

    # deals with victory
    endTime = round(time.time() - startTime)
    print('You\'ve sunk all of my ships with {} guesses in {} seconds. You win!'.format(guesses, endTime))

# start the game
main()

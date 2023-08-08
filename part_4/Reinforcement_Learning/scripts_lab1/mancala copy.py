import numpy as np
import pandas as pd
from tabulate import tabulate
import warnings
import torch
warnings.simplefilter(action='ignore', category=FutureWarning) # Подавляем предумпреждение о сравнении типов


class Kalah:
    '''
    Класс, где реализуется игра Калах.
    При создании класса в конструкторе заполняется state (места хранения камней)
    '''
    __state: torch.Tensor
    __is_game_over: bool = False
    __player_making_step: int = 1
    __device: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    __winner: int


    def __init__(self) -> None:
        l = torch.arange(0, 14)
        self.__state = torch.where(
            (l < 6) | (l > 7),
            6, 0
        )
        self.__state.to(self.__device)


    def take_step(self, id_heap: int) -> str:

        if self.__is_game_over:
            return "Игра окончена! Обновите игру"

        if id_heap not in range(1, 7):
            return "Неверно указана куча! Попробуйте снова!"

        if self.__state[id_heap - 1] == 0:
            return "Куча пустая, выберите другую!"

        temp = int(self.__state[id_heap-1])
        self.__state[id_heap-1] = 0

        cursor = id_heap - 1
        while temp > 0:
            temp -= 1
            if cursor in range(0, 6):
                cursor += 1
            elif cursor == 6:
                cursor = 13
            elif cursor in range(9, 14):
                cursor -= 1
            elif cursor == 8:
                cursor = 0

            self.__state[cursor] += 1

        captured = ""
        additional_step = ""
        if cursor < 6 and self.__state[cursor] == 1 and self.__state[cursor%6-6] != 0:
            captured = " Захват!"
            self.__state[6] += sum([self.__state[cursor], self.__state[cursor%6-6]])
            self.__state[cursor], self.__state[cursor%6-6] = 0., 0.
        elif cursor == 6:
            additional_step = "Дополнительный ход!"

        if cursor != 6:
            # Переворачиваем поле для хода второго игрока
            self.__state = torch.flip(self.__state, [0])

            # Смена игрока
            self.__player_making_step = 3 - self.__player_making_step

        # if all([x == 0 for x in self.__state[:6]]) or all([x == 0 for x in self.__state[8:]]):
        if not any(self.__state[:6]) or not any(self.__state[8:]):
            self.__is_game_over = True
            self.__winner = self.get_winner_in_end(True)
        elif self.__state[6] > 36 or self.__state[7] > 36:
            self.__is_game_over = True
            self.__winner = self.get_winner_in_end(False)

        return f"Хороший ход! {captured} {additional_step}"


    def set_new_game(self) -> None:
        self.__init__()
        self.__is_game_over = False
        self.__player_making_step = 1
        self.__winner = None


    def set_state(self, state: torch.Tensor) -> None:
        self.__state = state.clone()


    def set_is_game_over(self, value: bool):
        self.__is_game_over = value


    def get_score_player(self, player_id: int):
        return self.get_general_state()[5 + player_id]


    def get_game_over(self) -> bool:
        return self.__is_game_over


    def get_winner(self) -> int:
        return self.__winner


    def get_point(self, is_empty_heap) -> tuple:
        if is_empty_heap:
            return sum(self.__state[:7]), sum(self.__state[7:])
        return self.__state[6], self.__state[7]


    def get_winner_in_end(self, is_empty_heap: bool) -> int:
        if self.__player_making_step == 2:
            self.__player_making_step = 1
            self.__state = torch.flip(self.__state, [0])
        p1_point, p2_point = self.get_point(is_empty_heap)
        self.__state[6], self.__state[7] = p1_point, p2_point
        self.__state[:6] = torch.zeros(6)
        self.__state[8:] = torch.zeros(6)
        if p1_point == p2_point: return 0
        return 1 if p1_point > p2_point else 2


    def get_general_state(self) -> torch.Tensor:
        if self.__player_making_step == 2:
            return torch.flip(self.__state, [0])
        return self.__state


    def get_state(self) -> torch.Tensor:
        return self.__state


    def print_state(self) -> None:
        print(type(self))
        state = self.get_general_state()

        df = pd.DataFrame({
            "Pl2" : ["", state[7], "", "Pl1"],
            "6 <---- 1": [state[8:].detach().numpy(), "", state[:6].detach().numpy(), "1 ----> 6"],
            "" : ["", state[6], "", ""],
        })
        print(tabulate(df, showindex=False, headers=df.columns, stralign='center'))
        print()


    def get_player_making_step(self):
        return self.__player_making_step


    def get_player_winner(self) -> str:
        return f"Player {self.__winner} is winner!" if self.__winner != 0 else f"Appears to be a tie, maesters!"


def main():

    game = Kalah()
    game.print_state()
    while not game.get_game_over():
        step = 0
        while step not in range(1, 7):
            try:
                step = int(input(f"Player {game.get_player_making_step()} Введите номер кучи (от 1 до 6): "))
                if step not in range(1, 7):
                    raise ValueError
            except ValueError:
                print("Неверный ввод! Вводите цифры из диапазона!")
        rezult_step = game.take_step(step)
        if "Хороший ход!" in rezult_step:
            print(rezult_step)
            game.print_state()
        else:
            print(rezult_step)
        # print(game.get_state())
    print(game.get_player_winner())


if __name__ == '__main__':
    main()
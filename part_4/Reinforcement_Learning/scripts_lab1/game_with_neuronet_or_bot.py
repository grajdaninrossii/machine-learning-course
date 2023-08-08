from mancala import Kalah
import torch
import pandas
import os
import sys, copy, random

MODEL_NAME = "./models_lab1/model_best_1.pt"
# print(os.listdir("./models_lab1"))


# Бот простой, рандомно выбирает не нулевые элементы
def do_simple_bot_step(state: torch.Tensor) -> int:
    nonzero_state_indexs = torch.nonzero(state[:6]).flatten()
    rez = nonzero_state_indexs[torch.randint(0, len(nonzero_state_indexs), (1,))[0]]
    return rez + 1

# Бот МАСТЕР  захвата
def do_prof_bot_step(state: torch.Tensor) -> int:
    nonzero_state_indexs = torch.nonzero(state[:6]).flatten()
    state_copy = state.clone()
    captured = []
    additional_moves = []
    for cursor in nonzero_state_indexs:
        cursor = int(cursor)
        n = int(cursor)
        temp = int(state_copy[cursor])
        state_copy[cursor] = 0
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

            state_copy[cursor] += 1

        if cursor == 6:
            additional_moves.append(n)

        if cursor < 6 and state_copy[cursor] == 1 and state_copy[cursor%6-6] != 0:
            state_copy[6] += sum([state_copy[cursor], state_copy[cursor%6-6]])
            captured.append((n, sum([state_copy[cursor], state_copy[cursor%6-6]])))
        state_copy = state.clone()


    if captured != []:
        rez_ind = sorted(captured, key=lambda x: x[1])
        rez = rez_ind[-1][0]
    elif additional_moves != []:
        rez = additional_moves[torch.randint(0, len(additional_moves), (1,))[0]]
    else:
        rez = nonzero_state_indexs[torch.randint(0, len(nonzero_state_indexs), (1,))[0]]
    return rez + 1


def minimax(player, depth, game, move, alpha, beta, allowed_moves):
        if depth == 0:
            # print(len(game.get_point(True)))
            l = game.get_general_state()
            return [sum(l[:7]), sum(l[7:])][player-1]

        # test_board = game.copy()
        test_board = copy.deepcopy(game)
        test_board.take_step(move + 1)

        maxi = (test_board.get_player_making_step() == player)

        move_options = allowed_moves.clone()
        best_move = -sys.maxsize if maxi else sys.maxsize

        for move_slot in move_options:
            current_value = minimax(
                player, depth - 1, test_board, move_slot, alpha, beta, move_options)

            if maxi:
                best_move = max(current_value, best_move)
                alpha = max(alpha, best_move)
            else:
                best_move = min(current_value, best_move)
                beta = min(beta, best_move)

            if beta <= alpha:
                return best_move

        return best_move


def do_minimax_step(game: Kalah, current_player) -> int:
    moves_and_scores = []
    nonzero_state_indexs = torch.nonzero(game.get_state()[:6]).flatten()
    max_depth = 6
    for move in nonzero_state_indexs:
        minimax_score = minimax(
            current_player, max_depth, game, move, -sys.maxsize, sys.maxsize, nonzero_state_indexs)
        moves_and_scores.append([move, minimax_score])

    scores = [item[1] for item in moves_and_scores]
    max_score = max(scores)

    potential_moves = []
    for move_and_score in moves_and_scores:
        if move_and_score[1] == max_score:
            potential_moves.append(move_and_score[0])

    return random.choice(potential_moves) + 1


def enter_int_data(info: str, start: int, end: int) -> int:
    choice = 0
    while True:
        choice = input(info)
        try:
            choice = int(choice)
        except ValueError:
            print('Ошибка! Введите цифру!')
            continue
        if choice in range(start, end):
            break
        else:
            print('Нет выбора под таким номером! Попробуйте снова')

    return choice


# Ход нейронки
def take_step_neuronet(game: Kalah, model: torch.nn.Sequential):
    old_player_making_step = game.get_player_making_step()
    # print("Ход нейронки! ", old_player_making_step == game.get_player_making_step())
    bad_choise = []
    while not game.get_game_over() and old_player_making_step == game.get_player_making_step():
        # Выбор хода
        probs = model(game.get_state().to(torch.float))
        action = probs.argmax()

        while action in bad_choise:
            probs[probs.argmax()] = 0
            action = probs.argmax()

            rezult_step = game.take_step(action + 1)
        rezult_step = game.take_step(action + 1)
        if rezult_step == "Куча пустая, выберите другую!":
            count_choisen_zero += 1
            print("Neuronet took step, but heap is empty", game.get_player_making_step(), "Был выбор", action + 1, rezult_step)
            bad_choise.append(action)
        else:
            print(f"Нейросеть делает ход! Выбор лунки {action + 1}")
            game.print_state()
            bad_choise = []
        print(rezult_step)



def take_user_step(game: Kalah):
    # print(game.get_general_state())
    old_player_making_step = game.get_player_making_step()
    while not game.get_game_over() and old_player_making_step == game.get_player_making_step():
        user_step = enter_int_data(
            info=f"Player {game.get_player_making_step()} Введите номер кучи (от 1 до 6): ",
            start=1,
            end=7
        )
        rezult_step = game.take_step(user_step)
        if "Хороший ход!" in rezult_step:
            print(rezult_step)
            game.print_state()
        else:
            print(rezult_step)



def play_with_neuronet(game: Kalah, first_walker: int):
    # Загружаем модель
    model = torch.load(MODEL_NAME)
    while not game.get_game_over():
        if first_walker != game.get_player_making_step():
            take_step_neuronet(game, model)
        else:
            # print(game.get_general_state())
            take_user_step(game)
    else:
        print(game.get_player_winner())


def take_step_bot(game: Kalah, minmax = True):
    old_player_making_step = game.get_player_making_step()
    while not game.get_game_over() and old_player_making_step == game.get_player_making_step():
        # Выбор режима для бота
        # bot_action = do_prof_bot_step(game.get_state())
        # bot_action = do_simple_bot_step(game.get_state())
        # bot_action = 0
        if minmax:
            bot_action = do_minimax_step(game, game.get_player_making_step())
            print(game.take_step(bot_action))
            game.print_state()
            # print('kek')


def play_with_bot(game: Kalah, first_walker: int):
    while not game.get_game_over():
        if first_walker != game.get_player_making_step():
            take_step_bot(game)
        else:
            take_user_step(game)
    else:
        print(game.get_player_winner())


def play_game(oponent: int, first_walker: int):
    game = Kalah()
    game.print_state()
    if oponent == 1:
        play_with_bot(game, first_walker)
    else:
        play_with_neuronet(game, first_walker)


def main():
    # oponent = enter_int_data(
    #     info="Типы игры:\n1 - с ботом\n2 - с нейронкой\nВыберите тип игры: ",
    #     start=1,
    #     end=3
    # )
    first_walker = enter_int_data(
        info="Кто первый ходит?\n1 - ваш ход первый\n2 - ход опонента первый\nВыберите ход: ",
        start=1,
        end=3
    )
    play_game(2, first_walker)



if __name__ == "__main__":
    main()
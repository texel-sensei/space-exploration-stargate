import os.path
import importlib

import pygame


def main() -> int:
    pygame.init()

    # import must happen after pygame.init(), so we can use pygame stuff on module level
    from . import logic

    running = True

    screen = pygame.display.set_mode(
        (1000, 1000), vsync=1, flags=pygame.RESIZABLE | pygame.SCALED
    )
    clock = pygame.time.Clock()
    has_working_module = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill("black")

        match reload_module(logic):
            case False:
                has_working_module = False
            case True:
                has_working_module = True

        if has_working_module:
            try:
                logic.update(screen)
            except Exception as e:
                print(e)
                has_working_module = False

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()
    return 0


def reload_module(module, cache={}) -> bool:
    path = module.__file__
    try:
        changed = os.path.getmtime(path)
    except FileNotFoundError:
        return

    if path not in cache:
        cache[path] = changed
    else:
        old_time = cache[module.__file__]
        if changed > old_time:
            cache[module.__file__] = changed
            try:
                importlib.reload(module)
                return True
            except Exception as e:
                print(e)
                return False
    return None

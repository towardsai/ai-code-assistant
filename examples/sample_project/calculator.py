"""A tiny sample module the agent and tests can explore."""


class Calculator:
    """A minimal calculator with a running total."""

    def __init__(self):
        self.total = 0

    def add(self, x):
        self.total += x
        return self.total

    def subtract(self, x):
        self.total -= x
        return self.total

    def reset(self):
        self.total = 0
        return self.total


def process_payment(amount, balance):
    """Deduct amount from balance if funds are sufficient."""
    if amount > balance:
        raise ValueError("insufficient funds")
    return balance - amount

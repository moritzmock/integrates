"""Choose one equally weighted team identity for a new sync commit."""

import secrets


TEAM_MEMBERS = (
    ("moritz-mock", "momock@unibz.it"),
    ("daniel-planotscher", "dani.plan@yahoo.de"),
    ("jonas-rabensteiner", "j.rabensteiner@icon.bz.it"),
    ("thomas-borsani", "tborsani@hpclgn02.unibz.it"),
)


def choose_author() -> tuple[str, str]:
    return secrets.choice(TEAM_MEMBERS)


def main() -> None:
    print(*choose_author(), sep="\t")


if __name__ == "__main__":
    main()

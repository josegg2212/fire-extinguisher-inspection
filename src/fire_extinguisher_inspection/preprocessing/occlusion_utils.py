"""Utilidades para crear oclusiones semi-sinteticas en crops."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal


ColorRGB = tuple[int, int, int]
TipoOclusion = Literal["partial", "blocked"]

COLORES_OCLUSION: tuple[ColorRGB, ...] = (
    (33, 37, 41),
    (75, 85, 99),
    (120, 113, 108),
    (151, 124, 87),
    (181, 166, 132),
    (229, 229, 220),
)


@dataclass(frozen=True)
class RegionOclusion:
    """Region rectangular base usada para dibujar una oclusion."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def ancho(self) -> int:
        return self.x2 - self.x1

    @property
    def alto(self) -> int:
        return self.y2 - self.y1


def _asegurar_pillow() -> tuple[object, object]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("No se puede importar Pillow. Instala requirements.txt.") from exc
    return Image, ImageDraw


def _limitar_color(valor: int) -> int:
    return max(0, min(255, valor))


def _variar_color(color: ColorRGB, rng: random.Random, rango: int = 18) -> ColorRGB:
    return tuple(_limitar_color(canal + rng.randint(-rango, rango)) for canal in color)


def elegir_region_oclusion(
    ancho: int,
    alto: int,
    porcentaje_area: float,
    rng: random.Random,
    *,
    desde_borde: bool = False,
) -> RegionOclusion:
    """Elige una region de oclusion reproducible y no siempre centrada."""

    if ancho <= 1 or alto <= 1:
        raise ValueError("La imagen debe tener ancho y alto mayores que 1.")

    porcentaje_area = max(0.05, min(0.92, porcentaje_area))
    area_objetivo = ancho * alto * porcentaje_area

    if desde_borde:
        borde = rng.choice(("left", "right", "top", "bottom"))
        if borde in {"left", "right"}:
            occ_ancho = max(1, min(ancho - 1, int(ancho * porcentaje_area)))
            if borde == "left":
                return RegionOclusion(0, 0, occ_ancho, alto)
            return RegionOclusion(ancho - occ_ancho, 0, ancho, alto)

        occ_alto = max(1, min(alto - 1, int(alto * porcentaje_area)))
        if borde == "top":
            return RegionOclusion(0, 0, ancho, occ_alto)
        return RegionOclusion(0, alto - occ_alto, ancho, alto)

    relacion_aspecto = rng.uniform(0.45, 2.2)
    occ_ancho = max(1, int((area_objetivo * relacion_aspecto) ** 0.5))
    occ_alto = max(1, int(area_objetivo / max(occ_ancho, 1)))

    occ_ancho = min(occ_ancho, max(1, ancho - 1))
    occ_alto = min(occ_alto, max(1, alto - 1))
    x1 = rng.randint(0, max(0, ancho - occ_ancho))
    y1 = rng.randint(0, max(0, alto - occ_alto))
    return RegionOclusion(x1, y1, x1 + occ_ancho, y1 + occ_alto)


def _polygon_desde_region(region: RegionOclusion, rng: random.Random) -> list[tuple[int, int]]:
    jitter_x = max(2, region.ancho // 8)
    jitter_y = max(2, region.alto // 8)
    return [
        (region.x1 + rng.randint(-jitter_x, jitter_x), region.y1 + rng.randint(-jitter_y, jitter_y)),
        (region.x2 + rng.randint(-jitter_x, jitter_x), region.y1 + rng.randint(-jitter_y, jitter_y)),
        (region.x2 + rng.randint(-jitter_x, jitter_x), region.y2 + rng.randint(-jitter_y, jitter_y)),
        (region.x1 + rng.randint(-jitter_x, jitter_x), region.y2 + rng.randint(-jitter_y, jitter_y)),
    ]


def _dibujar_textura(draw: object, region: RegionOclusion, color: ColorRGB, rng: random.Random) -> None:
    """Anade lineas sutiles para evitar bloques completamente planos."""

    lineas = rng.randint(2, 5)
    for _ in range(lineas):
        if rng.random() < 0.5:
            y = rng.randint(region.y1, max(region.y1, region.y2 - 1))
            draw.line(
                (region.x1, y, region.x2, y + rng.randint(-3, 3)),
                fill=_variar_color(color, rng, 28),
                width=rng.randint(1, 3),
            )
        else:
            x = rng.randint(region.x1, max(region.x1, region.x2 - 1))
            draw.line(
                (x, region.y1, x + rng.randint(-3, 3), region.y2),
                fill=_variar_color(color, rng, 28),
                width=rng.randint(1, 3),
            )


def aplicar_oclusion(
    imagen: object,
    rng: random.Random,
    *,
    tipo: TipoOclusion,
) -> object:
    """Aplica una oclusion parcial o fuerte sobre una imagen PIL."""

    _, ImageDraw = _asegurar_pillow()
    resultado = imagen.convert("RGB").copy()
    ancho, alto = resultado.size

    if tipo == "partial":
        porcentaje_area = rng.uniform(0.20, 0.45)
        desde_borde = rng.random() < 0.25
    elif tipo == "blocked":
        porcentaje_area = rng.uniform(0.55, 0.85)
        desde_borde = rng.random() < 0.55
    else:
        raise ValueError(f"Tipo de oclusion no soportado: {tipo}")

    region = elegir_region_oclusion(ancho, alto, porcentaje_area, rng, desde_borde=desde_borde)
    color = _variar_color(rng.choice(COLORES_OCLUSION), rng)
    draw = ImageDraw.Draw(resultado)

    if rng.random() < 0.35:
        draw.polygon(_polygon_desde_region(region, rng), fill=color)
    else:
        draw.rectangle((region.x1, region.y1, region.x2, region.y2), fill=color)

    _dibujar_textura(draw, region, color, rng)
    return resultado


def aplicar_oclusion_parcial(imagen: object, rng: random.Random) -> object:
    """Crea una variante `partially_occluded` del crop."""

    return aplicar_oclusion(imagen, rng, tipo="partial")


def aplicar_oclusion_fuerte(imagen: object, rng: random.Random) -> object:
    """Crea una variante `blocked` del crop."""

    return aplicar_oclusion(imagen, rng, tipo="blocked")

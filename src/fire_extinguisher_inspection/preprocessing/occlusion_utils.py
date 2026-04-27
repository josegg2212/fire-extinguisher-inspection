"""Utilidades para crear oclusiones semi-sinteticas en crops."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal


ColorRGB = tuple[int, int, int]
TipoOclusion = Literal["partial", "blocked"]
BBox = tuple[int, int, int, int]

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


def _normalizar_bbox_objeto(bbox_objeto: BBox | None, ancho: int, alto: int) -> BBox | None:
    if bbox_objeto is None:
        return None

    x1, y1, x2, y2 = [int(valor) for valor in bbox_objeto]
    x1 = max(0, min(ancho - 1, x1))
    y1 = max(0, min(alto - 1, y1))
    x2 = max(x1 + 1, min(ancho, x2))
    y2 = max(y1 + 1, min(alto, y2))
    return x1, y1, x2, y2


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


def elegir_region_oclusion_contextual(
    ancho: int,
    alto: int,
    bbox_objeto: BBox,
    rng: random.Random,
    *,
    tipo: TipoOclusion,
) -> RegionOclusion:
    """Elige una region que interacciona con el extintor dentro del crop contextual."""

    ox1, oy1, ox2, oy2 = bbox_objeto
    obj_ancho = max(1, ox2 - ox1)
    obj_alto = max(1, oy2 - oy1)

    if tipo == "partial":
        cobertura = rng.uniform(0.20, 0.45)
        escala_extra = rng.uniform(1.05, 1.45)
    else:
        cobertura = rng.uniform(0.55, 0.85)
        escala_extra = rng.uniform(1.20, 1.85)

    estrategia = rng.choice(("desde_borde", "panel_objeto", "franja"))
    if estrategia == "desde_borde":
        borde = rng.choice(("left", "right", "top", "bottom"))
        if borde == "left":
            x1 = 0
            x2 = min(ancho, int(round(ox1 + obj_ancho * cobertura * escala_extra)))
            y1 = max(0, int(round(oy1 - obj_alto * rng.uniform(0.25, 0.65))))
            y2 = min(alto, int(round(oy2 + obj_alto * rng.uniform(0.25, 0.65))))
        elif borde == "right":
            x1 = max(0, int(round(ox2 - obj_ancho * cobertura * escala_extra)))
            x2 = ancho
            y1 = max(0, int(round(oy1 - obj_alto * rng.uniform(0.25, 0.65))))
            y2 = min(alto, int(round(oy2 + obj_alto * rng.uniform(0.25, 0.65))))
        elif borde == "top":
            x1 = max(0, int(round(ox1 - obj_ancho * rng.uniform(0.25, 0.65))))
            x2 = min(ancho, int(round(ox2 + obj_ancho * rng.uniform(0.25, 0.65))))
            y1 = 0
            y2 = min(alto, int(round(oy1 + obj_alto * cobertura * escala_extra)))
        else:
            x1 = max(0, int(round(ox1 - obj_ancho * rng.uniform(0.25, 0.65))))
            x2 = min(ancho, int(round(ox2 + obj_ancho * rng.uniform(0.25, 0.65))))
            y1 = max(0, int(round(oy2 - obj_alto * cobertura * escala_extra)))
            y2 = alto
    elif estrategia == "franja":
        if rng.random() < 0.5:
            occ_ancho = max(4, int(round(obj_ancho * cobertura * escala_extra)))
            x_centro = rng.randint(ox1, max(ox1, ox2 - 1))
            x1 = max(0, x_centro - occ_ancho // 2)
            x2 = min(ancho, x1 + occ_ancho)
            y1 = max(0, int(round(oy1 - obj_alto * rng.uniform(0.35, 0.9))))
            y2 = min(alto, int(round(oy2 + obj_alto * rng.uniform(0.35, 0.9))))
        else:
            occ_alto = max(4, int(round(obj_alto * cobertura * escala_extra)))
            y_centro = rng.randint(oy1, max(oy1, oy2 - 1))
            y1 = max(0, y_centro - occ_alto // 2)
            y2 = min(alto, y1 + occ_alto)
            x1 = max(0, int(round(ox1 - obj_ancho * rng.uniform(0.35, 0.9))))
            x2 = min(ancho, int(round(ox2 + obj_ancho * rng.uniform(0.35, 0.9))))
    else:
        occ_ancho = max(4, int(round(obj_ancho * cobertura * escala_extra)))
        occ_alto = max(4, int(round(obj_alto * cobertura * escala_extra)))
        x_centro = rng.randint(ox1, max(ox1, ox2 - 1))
        y_centro = rng.randint(oy1, max(oy1, oy2 - 1))
        x1 = max(0, x_centro - occ_ancho // 2)
        y1 = max(0, y_centro - occ_alto // 2)
        x2 = min(ancho, x1 + occ_ancho)
        y2 = min(alto, y1 + occ_alto)

    if x2 <= x1:
        x2 = min(ancho, x1 + 1)
    if y2 <= y1:
        y2 = min(alto, y1 + 1)
    return RegionOclusion(x1, y1, x2, y2)


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


def _dibujar_ruido_sutil(draw: object, region: RegionOclusion, color: ColorRGB, rng: random.Random) -> None:
    """Anade puntos y pequenas variaciones que rompen la uniformidad del obstaculo."""

    total_puntos = max(8, min(80, (region.ancho * region.alto) // 900))
    for _ in range(total_puntos):
        x = rng.randint(region.x1, max(region.x1, region.x2 - 1))
        y = rng.randint(region.y1, max(region.y1, region.y2 - 1))
        radio = rng.randint(1, 2)
        draw.ellipse(
            (x - radio, y - radio, x + radio, y + radio),
            fill=_variar_color(color, rng, 34),
        )


def aplicar_oclusion(
    imagen: object,
    rng: random.Random,
    *,
    tipo: TipoOclusion,
    bbox_objeto: BBox | None = None,
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

    bbox_normalizada = _normalizar_bbox_objeto(bbox_objeto, ancho, alto)
    if bbox_normalizada is not None:
        region = elegir_region_oclusion_contextual(ancho, alto, bbox_normalizada, rng, tipo=tipo)
    else:
        region = elegir_region_oclusion(ancho, alto, porcentaje_area, rng, desde_borde=desde_borde)

    color = _variar_color(rng.choice(COLORES_OCLUSION), rng)
    draw = ImageDraw.Draw(resultado)

    forma = rng.choice(("rectangulo", "poligono", "franja"))
    if forma == "poligono":
        draw.polygon(_polygon_desde_region(region, rng), fill=color)
    elif forma == "franja":
        inclinacion = rng.randint(-region.alto // 4, region.alto // 4)
        puntos = [
            (region.x1, region.y1 + inclinacion),
            (region.x2, region.y1 - inclinacion),
            (region.x2, region.y2),
            (region.x1, region.y2),
        ]
        draw.polygon(puntos, fill=color)
    else:
        draw.rectangle((region.x1, region.y1, region.x2, region.y2), fill=color)

    _dibujar_textura(draw, region, color, rng)
    _dibujar_ruido_sutil(draw, region, color, rng)
    return resultado


def aplicar_oclusion_parcial(
    imagen: object,
    rng: random.Random,
    bbox_objeto: BBox | None = None,
) -> object:
    """Crea una variante `partially_occluded` del crop."""

    return aplicar_oclusion(imagen, rng, tipo="partial", bbox_objeto=bbox_objeto)


def aplicar_oclusion_fuerte(
    imagen: object,
    rng: random.Random,
    bbox_objeto: BBox | None = None,
) -> object:
    """Crea una variante `blocked` del crop."""

    return aplicar_oclusion(imagen, rng, tipo="blocked", bbox_objeto=bbox_objeto)

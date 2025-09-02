from flask import Flask, render_template, request
import numpy as np
import sympy as sp
import re

app = Flask(__name__)


def generar_g_automatica(f_expr, x0, local_dict):
    x = sp.symbols('x')
    g_opciones = []
    try:
        # Definimos la ecuación f(x) = 0
        eq = sp.Eq(f_expr, 0)

        # Intentamos resolver para x
        soluciones = sp.solve(eq, x, dict=True)

        # Guardamos las soluciones en formato latex legible
        for sol in soluciones:
            g_forma = sol[x]
            g_opciones.append({
                "expr": g_forma,
                "latex": sp.latex(g_forma),
                "sugerido_x0": float(x0) if x0 is not None else 1
            })

        # Si no encontró nada (ejemplo x^3 - x - 1), damos sugerencias manuales
        if not g_opciones:
            try:
                # Caso típico: x^3 - x - 1 => x = (x+1)^(1/3)
                g_forma = (x + 1)**(sp.Rational(1, 3))
                g_opciones.append({
                    "expr": g_forma,
                    "latex": sp.latex(g_forma),
                    "sugerido_x0": float(x0) if x0 is not None else 1
                })
            except:
                pass

    except Exception as e:
        print("Error en generar_g_automatica:", e)

    return g_opciones


# Métodos numéricos
def punto_fijo(g, x0, tol, max_iter):
    resultados = []
    x = x0
    for i in range(max_iter):
        x_new = g(x)
        # Ya no se valida si el resultado es complejo
        error = abs(x_new - x)
        resultados.append({'iter': i+1, 'valor': x_new, 'error': error})
        if error < tol:
            return x_new, resultados
        x = x_new
    return None, resultados

def newton_raphson(f, df, x0, tol, max_iter):
    resultados = []
    x = x0
    for i in range(max_iter):
        f_x = f(x)
        df_x = df(x)
        if df_x == 0:
            break
        x_new = x - f_x / df_x
        if isinstance(x_new, complex):
            raise ValueError('Se obtuvo un número complejo en la iteración {}. Verifique la función f(x) y el valor inicial.'.format(i+1))
        # Error relativo porcentual
        eak = abs((x_new - x) / x_new) * 100 if x_new != 0 else 0
        resultados.append({
            'iter': i+1,
            'xk': x,
            'fxk': f_x,
            'dfxk': df_x,
            'xk1': x_new,
            'eak': eak
        })
        if abs(x_new - x) < tol:
            return x_new, resultados
        x = x_new
    return None, resultados

def secante(f, x0, x1, tol, max_iter):
    resultados = []
    xk_1 = x0
    xk = x1
    for i in range(max_iter):
        fxk_1 = f(xk_1)
        fxk = f(xk)
        if fxk - fxk_1 == 0:
            break
        xk1 = xk - fxk * (xk - xk_1) / (fxk - fxk_1)
        if isinstance(xk1, complex):
            raise ValueError('Se obtuvo un número complejo en la iteración {}. Verifique la función f(x) y los valores iniciales.'.format(i+1))
        eak = abs((xk1 - xk) / xk1) * 100 if xk1 != 0 else 0
        resultados.append({
            'iter': i+1,
            'xk_1': xk_1,
            'xk': xk,
            'fxk_1': fxk_1,
            'fxk': fxk,
            'xk1': xk1,
            'eak': eak
        })
        if abs(xk1 - xk) < tol:
            return xk1, resultados
        xk_1, xk = xk, xk1
    return None, resultados

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    iteraciones = []
    error_msg = None
    parametros = {}
    if request.method == 'POST':
        metodo = request.form['metodo']
        func_str = request.form['func_str']
        x0 = request.form.get('x0', None)
        tol = request.form.get('tol', None)
        max_iter = request.form.get('max_iter', None)
        # Validar campos numéricos
        try:
            x0 = float(x0) if x0 is not None else None
            tol = float(tol) if tol is not None else None
            max_iter = int(max_iter) if max_iter is not None else None
        except Exception:
            error_msg = 'Error en los valores numéricos. Verifique los datos.'
            return render_template(
                        'index.html',
                        resultado=resultado,
                        iteraciones=iteraciones,
                        error_msg=error_msg,
                        g_expr=parametros.get("g_expr") if parametros else None
                    )
        # Permitir parámetros personalizados
        param_str = request.form.get('parametros', '')
        if param_str:
            try:
                for p in param_str.split(','):
                    k, v = p.split('=')
                    parametros[k.strip()] = float(v.strip())
            except Exception:
                error_msg = 'Error en los parámetros. Use el formato: a=2, b=3'
                return render_template('index.html', resultado=resultado, iteraciones=iteraciones, error_msg=error_msg)
        # Procesar símbolos matemáticos comunes y LaTeX básico en la expresión
        def procesar_expr(expr):
            # Eliminar delimitadores LaTeX
            expr = expr.replace('$$', '')
            expr = expr.replace('$', '')
            # Reemplazos básicos de LaTeX a sympy
            expr = expr.replace('^', '**')
            expr = expr.replace('\\sqrt', 'sqrt')
            expr = expr.replace('\\pi', 'pi')
            expr = expr.replace('\\sin', 'sin')
            expr = expr.replace('\\cos', 'cos')
            expr = expr.replace('\\tan', 'tan')
            expr = expr.replace('\\ln', 'log')
            expr = expr.replace('\\log', 'log')
            expr = expr.replace('{', '(').replace('}', ')')
            expr = expr.replace('sen', 'sin')
            expr = expr.replace('tg', 'tan')
            expr = expr.replace('√', 'sqrt')
            expr = expr.replace('π', 'pi')
            # Solo insertar * entre número y variable, y entre variable y paréntesis
            expr = re.sub(r'(\d+)\s*x', r'\1*x', expr)
            expr = re.sub(r'([a-zA-Z0-9])\s*\(', r'\1*(', expr)
            # Eliminar espacios innecesarios
            expr = expr.replace(' ', '')
            # 1. Manejar e^x o 2e^x -> exp(x) o 2*exp(x)
            expr = re.sub(r'(\d*)e\*\*(\d+)', r'\1*exp(\2)', expr)
            expr = re.sub(r'(\d*)e\^(\(?[^\)]+\)?)', r'\1*exp(\2)', expr)
            expr = re.sub(r'\be\b', 'exp(1)', expr)

            # 2. Número seguido de variable o función: 2x -> 2*x, 3sin(x) -> 3*sin(x)
            expr = re.sub(r'(\d)([a-zA-Z(])', r'\1*\2', expr)
            # 3. Variable o función seguida de paréntesis: x(x+1) -> x*(x+1), sin(x)(x+1) -> sin(x)*(x+1)
            expr = re.sub(r'([a-zA-Z0-9\)])(\()', r'\1*\2', expr)
            # 4. Paréntesis seguidos de variable o función: (x+1)x -> (x+1)*x
            expr = re.sub(r'(\))([a-zA-Z(])', r'\1*\2', expr)
            # (Revertido: no agregar * entre función y variable)
            return expr
        func_str = procesar_expr(func_str)
        x = sp.symbols('x')
        # Usar funciones simbólicas de SymPy para sympify
        sympy_dict = {**parametros, 'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan, 'exp': sp.exp, 'log': sp.log, 'sqrt': sp.sqrt, 'pi': sp.pi, 'e': sp.E}
        # Usar funciones numéricas de numpy para lambdify
        numpy_dict = {**parametros, 'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 'pi': np.pi, 'e': np.e}
        try:
            f_expr = sp.sympify(func_str, locals=sympy_dict)
            f = sp.lambdify(x, f_expr, modules=['numpy', numpy_dict])
            if metodo == 'punto_fijo':
                f_expr = sp.sympify(func_str, locals=sympy_dict)
                opciones_g = generar_g_automatica(f_expr, x0, sympy_dict)
                g_idx = request.form.get('g_idx', None)
                if g_idx is not None and g_idx != '':
                    try:
                        g_idx = int(g_idx)
                        g_expr = opciones_g[g_idx]['expr']
                        g_func = sp.lambdify(x, g_expr, modules=['numpy', numpy_dict])
                        g_prime = sp.lambdify(x, sp.diff(g_expr, x), modules=['numpy', numpy_dict])
                        if abs(g_prime(x0)) >= 1:
                            error_msg = f"g(x) no converge para x₀={x0}. Sugerencia: prueba x₀={opciones_g[g_idx]['sugerido_x0']} o elige otra forma de g(x)."
                            raiz = None
                            iteraciones = []
                        else:
                            raiz, iteraciones = punto_fijo(g_func, x0, tol, max_iter)
                            parametros["g_expr"] = str(g_expr)
                    except Exception as e:
                        error_msg = f"Error al procesar g(x): {e}"
                        raiz = None
                        iteraciones = []
                else:
                    raiz = None
                    iteraciones = []
                    parametros["g_opciones"] = [{'latex': g['latex'], 'sugerido_x0': g['sugerido_x0']} for g in opciones_g]
            elif metodo == 'newton':
                df_expr = sp.diff(sp.sympify(func_str, locals=sympy_dict), x)
                df = sp.lambdify(x, df_expr, modules=['numpy', numpy_dict])
                raiz, iteraciones = newton_raphson(f, df, x0, tol, max_iter)
            elif metodo == 'secante':
                x0 = request.form.get('x0', None)
                x1 = request.form.get('x1', None)
                try:
                    x0 = float(x0) if x0 is not None else None
                    x1 = float(x1) if x1 is not None else None
                except Exception:
                    error_msg = 'Error en x₀ o xₖ. Verifique los datos.'
                    return render_template('index.html', resultado=resultado, iteraciones=iteraciones, error_msg=error_msg)
                if x0 is None or x1 is None:
                    error_msg = 'Debes ingresar ambos valores: xₖ₋₁ y xₖ.'
                    return render_template('index.html', resultado=resultado, iteraciones=iteraciones, error_msg=error_msg)
                raiz, iteraciones = secante(f, x0, x1, tol, max_iter)
            else:
                error_msg = 'Método no válido.'
                raiz = None
            resultado = raiz
        except ValueError as ve:
            error_msg = str(ve)
            resultado = None
            iteraciones = []
            return render_template('index.html', resultado=resultado, iteraciones=iteraciones, error_msg=error_msg)
        except Exception as e:
            error_msg = f'Error en la ecuación: {e}'
            resultado = None
            iteraciones = []
            return render_template('index.html', resultado=resultado, iteraciones=iteraciones, error_msg=error_msg)
    return render_template(
        'index.html',
        resultado=resultado,
        iteraciones=iteraciones,
        error_msg=error_msg,
        g_expr=parametros.get("g_expr") if parametros else None,
        g_opciones=parametros.get("g_opciones") if parametros else None
    )


if __name__ == '__main__':
    app.run(debug=True)
